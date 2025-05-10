import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# --- Data Preparation ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.seq_len = seq_len
        self.tokens = []
        for txt in texts:
            toks = tokenizer(txt)
            # pad or truncate
            for i in range(0, len(toks) - seq_len):
                self.tokens.append(toks[i:i+seq_len+1])
        self.tokens = np.array(self.tokens)
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        seq = self.tokens[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[-1], dtype=torch.long)

# --- EM-based "Transformer" Block ---
class EMAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, top_k=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.top_k = top_k  # for sparsity

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, _ = x.shape
        Q = self.Wq(x).view(B, L, self.num_heads, self.d_head)
        K = self.Wk(x).view(B, L, self.num_heads, self.d_head)
        V = self.Wv(x).view(B, L, self.num_heads, self.d_head)

        # E-step: responsibilities = attention weights
        # compute raw scores
        scores = torch.einsum('bihd,bjhd->bijh', Q, K) / np.sqrt(self.d_head)  # (B, L, L, H)
        weights = torch.softmax(scores, dim=2)
        # sparsity: keep only top_k weights per query if specified
        if self.top_k is not None:
            topk_vals, topk_idx = weights.topk(self.top_k, dim=2)
            mask = torch.zeros_like(weights)
            mask.scatter_(2, topk_idx, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

        # M-step: value aggregation (could re-fit Wv here via weighted least squares)
        attn_out = torch.einsum('bijh,bjhd->bihd', weights, V)
        attn_out = attn_out.contiguous().view(B, L, -1)
        return self.out(attn_out)

class EMFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, top_k=None):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.top_k = top_k

    def forward(self, x):
        # E-step: gating over hidden units
        attn_scores = self.W1(x)  # (B,L,d_ff)
        weights = torch.softmax(attn_scores, dim=-1)
        # sparsity: keep top_k units
        if self.top_k is not None:
            topk_vals, topk_idx = weights.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(-1, topk_idx, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # M-step: expert aggregation (could re-fit W1/W2 via weighted least squares)
        mixture = torch.einsum('bils,blds->bids', weights, self.W2.weight.unsqueeze(0).unsqueeze(0).expand(weights.size(0), weights.size(1), -1, -1))
        return mixture

# --- Full EM-Transformer Stub ---
class EMTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, seq_len, top_k_attn=None, top_k_ff=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': EMAttentionBlock(d_model, num_heads, top_k_attn),
                'ffn': EMFeedForwardBlock(d_model, d_ff, top_k_ff),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
            }))
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        x = self.embed(x) + self.pos_embed[:, :L]
        for layer in self.layers:
            attn_out = layer['attn'](x)
            x = layer['ln1'](x + attn_out)
            ffn_out = layer['ffn'](x)
            x = layer['ln2'](x + ffn_out)
        return self.to_logits(x[:, -1, :])

# --- Sequence Generation ---
def generate(model, tokenizer, prompt, max_len):
    model.eval()
    tokens = tokenizer(prompt)
    for _ in range(max_len):
        inp = torch.tensor(tokens[-model.pos_embed.size(1):]).unsqueeze(0)
        logits = model(inp)
        next_tok = torch.argmax(logits, dim=-1).item()
        tokens.append(next_tok)
    return tokenizer.decode(tokens)

# --- Areas to Promote Sparsity ---
# 1. In EMAttentionBlock: use top_k to limit attention span (approximate sparse attention).
# 2. In EMFeedForwardBlock: use top_k to limit active hidden units (sparse experts).
# 3. During E-step: keep only top responsibilities per leaf/expert.
# 4. Implement hierarchical routing (trees) to reduce per-token cost from O(L) to O(depth).
# 5. Use streaming I/O (DuckDB) to compute sufficient stats for large datasets.

# This skeleton sets up the core EM-like blocks. You can later:
# - Add M-step routines to re-fit Wq/Wk/Wv via weighted least squares.
# - Replace fixed MLP embeddings with random feature maps.
# - Use DuckDB or mini-batch EM to handle datasets that exceed memory.
