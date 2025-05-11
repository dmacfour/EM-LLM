import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import time
import os
import json
from tqdm import tqdm
from collections import defaultdict
import math
import gc

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Config ---
class ModelConfig:
    def __init__(self, 
                 vocab_size=50257,  # GPT-2 vocab size
                 context_length=512,
                 embedding_dim=256,
                 output_dim=256,
                 num_trees=16,
                 tree_depth=6,
                 leaf_experts_dim=128,
                 expert_count=8,
                 batch_size=32,
                 learning_rate=1e-4,
                 weight_decay=0.01,
                 top_k_routing=4,
                 tree_granularity_levels=["coarse", "medium", "fine"],
                 max_training_steps=10000,
                 warmup_steps=500,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir='./checkpoints',
                 model_name='em_tree_llm'):
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.leaf_experts_dim = leaf_experts_dim
        self.expert_count = expert_count
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.top_k_routing = top_k_routing
        self.tree_granularity_levels = tree_granularity_levels
        self.max_training_steps = max_training_steps
        self.warmup_steps = warmup_steps
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        
        # Derived parameters
        self.max_leaves = 2**tree_depth
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config

# --- Data Preparation ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []
        
        # Tokenize all texts
        print("Tokenizing texts...")
        for text in tqdm(texts):
            tokens = tokenizer.encode(text)
            # Create windows of tokens
            for i in range(0, max(1, len(tokens) - seq_len)):
                self.examples.append(tokens[i:i+seq_len+1])
                
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Input is all tokens except the last one
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        # Target is all tokens except the first one
        y = torch.tensor(tokens[-1], dtype=torch.long)
        return x, y

# --- Tree-Based Routing ---
class SoftDecisionNode(nn.Module):
    """Soft decision node for probabilistic routing"""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # Binary decision: left or right
        )
    
    def forward(self, x):
        """Returns probability of routing left vs right"""
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        return probs

class LeafExpert(nn.Module):
    """Expert model at each leaf node"""
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim or input_dim * 2
        
        # Expert model (2-layer MLP with residual connection)
        self.expert = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, output_dim)
        )
        
        # For closed-form updates in EM
        self.regularization = 1e-5
        self.accumulated_stats = {
            'XTX': None,  # X^T W X
            'XTy': None,  # X^T W y
            'weight_sum': 0.0  # Sum of sample weights
        }
    
    def forward(self, x):
        return self.expert(x)
    
    def update_statistics(self, x, y, weight):
        """Accumulate sufficient statistics for closed-form update"""
        # Reshape inputs if needed
        if len(x.shape) == 2:  # (batch_size, feature_dim)
            pass
        elif len(x.shape) == 3:  # (batch_size, seq_len, feature_dim)
            x = x.view(-1, x.shape[-1])
            y = y.view(-1, y.shape[-1] if len(y.shape) > 2 else 1)
            weight = weight.view(-1, 1)
        
        # Apply weights
        weighted_x = x * torch.sqrt(weight)
        weighted_y = y * torch.sqrt(weight)
        
        # Compute statistics
        XTX = weighted_x.T @ weighted_x
        XTy = weighted_x.T @ weighted_y
        weight_sum = weight.sum().item()
        
        # Accumulate statistics
        if self.accumulated_stats['XTX'] is None:
            self.accumulated_stats['XTX'] = XTX
            self.accumulated_stats['XTy'] = XTy
        else:
            self.accumulated_stats['XTX'] += XTX
            self.accumulated_stats['XTy'] += XTy
        
        self.accumulated_stats['weight_sum'] += weight_sum
    
    def closed_form_update(self):
        """Perform closed-form parameter update using accumulated statistics"""
        if self.accumulated_stats['weight_sum'] < 1.0:
            return False  # Not enough data
        
        # Get accumulated statistics
        XTX = self.accumulated_stats['XTX']
        XTy = self.accumulated_stats['XTy']
        
        # Add regularization
        reg_matrix = self.regularization * torch.eye(
            XTX.shape[0], device=XTX.device
        )
        
        try:
            # Solve for weights: (X^T W X + λI)^(-1) X^T W y
            solution = torch.linalg.solve(XTX + reg_matrix, XTy)
            
            # Update first layer weights
            with torch.no_grad():
                first_layer_size = self.expert[0].weight.shape[1]
                self.expert[0].weight.copy_(solution[:self.hidden_dim, :first_layer_size].T)
                
                # Update biases if they exist
                if hasattr(self.expert[0], 'bias') and self.expert[0].bias is not None:
                    self.expert[0].bias.copy_(solution[:self.hidden_dim, -1])
            
            # Reset statistics
            self.accumulated_stats = {
                'XTX': None,
                'XTy': None,
                'weight_sum': 0.0
            }
            
            return True
        
        except RuntimeError:
            print("Warning: Could not solve linear system for expert update.")
            return False

class ProbabilisticRoutingTree(nn.Module):
    """Tree that routes inputs to leaf experts using soft decisions"""
    def __init__(self, input_dim, output_dim, depth, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.num_leaves = 2**depth
        
        # Create decision nodes (binary tree has 2^d - 1 internal nodes)
        self.decision_nodes = nn.ModuleList([
            SoftDecisionNode(input_dim, hidden_dim) 
            for _ in range(2**depth - 1)
        ])
        
        # Create leaf experts
        self.leaf_experts = nn.ModuleList([
            LeafExpert(input_dim, output_dim)
            for _ in range(2**depth)
        ])
        
        # For storing routing probabilities
        self.last_routing_probs = None
    
    def compute_routing_probabilities(self, x):
        """Compute the probability of reaching each leaf node"""
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize routing probabilities for the root
        routing_probs = torch.ones(batch_size, 1, device=device)
        
        # List to collect leaf probabilities
        leaf_probs = torch.zeros(batch_size, self.num_leaves, device=device)
        
        # Queue for breadth-first traversal
        queue = [(0, 0, routing_probs)]  # (node_idx, depth, probabilities)
        
        while queue:
            node_idx, node_depth, probs = queue.pop(0)
            
            if node_depth == self.depth:
                # We've reached a leaf
                leaf_idx = node_idx - (2**self.depth - 1)
                leaf_probs[:, leaf_idx] = probs.squeeze()
                continue
            
            # Get decision probabilities
            decision = self.decision_nodes[node_idx](x)  # (batch_size, 2)
            
            # Compute probabilities for children
            left_prob = probs * decision[:, 0].unsqueeze(1)
            right_prob = probs * decision[:, 1].unsqueeze(1)
            
            # Add children to queue
            left_idx = 2 * node_idx + 1
            right_idx = 2 * node_idx + 2
            queue.append((left_idx, node_depth + 1, left_prob))
            queue.append((right_idx, node_depth + 1, right_prob))
        
        # Store for potential M-step updates
        self.last_routing_probs = leaf_probs
        
        return leaf_probs
    
    def forward(self, x, top_k=None):
        """Forward pass through the tree, optionally using only top-k leaves"""
        # Compute routing probabilities
        leaf_probs = self.compute_routing_probabilities(x)  # (batch_size, num_leaves)
        
        # Apply top-k sparsity if requested
        if top_k is not None and top_k < self.num_leaves:
            top_k_values, top_k_indices = torch.topk(leaf_probs, k=top_k, dim=1)
            sparse_probs = torch.zeros_like(leaf_probs)
            sparse_probs.scatter_(1, top_k_indices, top_k_values)
            # Renormalize
            sparse_probs = sparse_probs / (sparse_probs.sum(dim=1, keepdim=True) + 1e-8)
            leaf_probs = sparse_probs
        
        # Apply leaf experts
        leaf_outputs = torch.zeros(
            x.shape[0], self.output_dim, device=x.device
        )
        
        # Only process leaves with significant probability
        for i, expert in enumerate(self.leaf_experts):
            # Get the probabilities for this leaf
            leaf_prob = leaf_probs[:, i].unsqueeze(1)
            
            # Skip leaves with negligible probability
            if leaf_prob.max().item() < 1e-3:
                continue
            
            # Compute expert output and weight by routing probability
            expert_output = expert(x)
            weighted_output = expert_output * leaf_prob
            
            # Aggregate outputs
            leaf_outputs += weighted_output
        
        return leaf_outputs, leaf_probs
    
    def m_step_update(self, x, y, responsibilities=None):
        """Update expert parameters using closed-form solution"""
        if responsibilities is None:
            responsibilities = self.last_routing_probs
        
        if responsibilities is None:
            return False  # No responsibilities available
        
        # Update each expert with weighted samples
        updated = 0
        for i, expert in enumerate(self.leaf_experts):
            # Get responsibilities for this expert
            weights = responsibilities[:, i].unsqueeze(1)
            
            # Skip experts with negligible responsibility
            if weights.max().item() < 1e-3:
                continue
            
            # Accumulate statistics
            expert.update_statistics(x, y, weights)
            
            # Perform closed-form update
            if expert.closed_form_update():
                updated += 1
        
        return updated > 0

class MultiResolutionForest(nn.Module):
    """Forest of trees with different granularity levels"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create trees with different granularities
        self.trees = nn.ModuleDict()
        
        for level in config.tree_granularity_levels:
            # Configure tree parameters based on granularity
            if level == "coarse":
                depth, num_trees = max(2, config.tree_depth - 2), max(2, config.num_trees // 4)
                hidden = config.embedding_dim // 2
            elif level == "medium":
                depth, num_trees = config.tree_depth, config.num_trees
                hidden = config.embedding_dim
            else:  # fine
                depth, num_trees = min(8, config.tree_depth + 2), max(2, config.num_trees * 2)
                hidden = config.embedding_dim * 2
            
            self.trees[level] = nn.ModuleList([
                ProbabilisticRoutingTree(
                    config.embedding_dim, 
                    config.output_dim,
                    depth,
                    hidden_dim=hidden
                ) for _ in range(num_trees)
            ])
        
        # Meta-model to combine predictions from different granularity levels
        self.meta_combiner = nn.Linear(len(config.tree_granularity_levels), 1)
        
        # Initialize with equal weights
        with torch.no_grad():
            self.meta_combiner.weight.fill_(1.0 / len(config.tree_granularity_levels))
            self.meta_combiner.bias.fill_(0.0)
    
    def forward(self, x):
        # Results from different granularity levels
        level_outputs = {}
        level_confidences = {}
        
        # Process each granularity level
        for level, trees in self.trees.items():
            level_results = []
            level_confs = []
            
            # Process each tree in this level
            for tree in trees:
                # Get outputs and routing probabilities
                tree_output, leaf_probs = tree(x, top_k=self.config.top_k_routing)
                
                # Use max responsibility as confidence measure
                confidence = leaf_probs.max(dim=1)[0].unsqueeze(1)  # (batch_size, 1)
                
                level_results.append(tree_output)
                level_confs.append(confidence)
            
            # Stack results from all trees in this level
            stacked_results = torch.stack(level_results, dim=1)  # (batch_size, num_trees, output_dim)
            stacked_confs = torch.stack(level_confs, dim=1)  # (batch_size, num_trees, 1)
            
            # Normalize confidences within this level
            normalized_confs = stacked_confs / (stacked_confs.sum(dim=1, keepdim=True) + 1e-8)
            
            # Weighted average of tree outputs based on confidence
            level_output = torch.sum(stacked_results * normalized_confs, dim=1)
            level_confidence = stacked_confs.mean(dim=1)
            
            level_outputs[level] = level_output
            level_confidences[level] = level_confidence
        
        # Combine outputs from different granularity levels
        all_outputs = torch.stack([level_outputs[level] for level in self.config.tree_granularity_levels], dim=1)
        all_confs = torch.stack([level_confidences[level] for level in self.config.tree_granularity_levels], dim=1)
        
        # Meta-combination weights
        meta_weights = F.softmax(self.meta_combiner(all_confs.squeeze(-1)), dim=1).unsqueeze(-1)
        
        # Final weighted output
        final_output = torch.sum(all_outputs * meta_weights, dim=1)
        
        return final_output

# --- Complete EM-Tree Language Model ---
class EMTreeLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Position embeddings
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.context_length, config.embedding_dim)
        )
        
        # Multi-resolution forest for token prediction
        self.forest = MultiResolutionForest(config)
        
        # Output projection to vocab
        self.output_projection = nn.Linear(config.output_dim, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(self, x):
        # Get sequence length
        _, seq_len = x.shape
        device = x.device
        
        # Embed tokens
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, emb_dim)
        
        # Add position embeddings
        pos_emb = self.position_embedding[:, :seq_len, :]
        x = token_emb + pos_emb  # (batch_size, seq_len, emb_dim)
        
        # Process the full sequence through the forest
        forest_output = self.forest(x)
        
        # Project to vocabulary
        logits = self.output_projection(forest_output)  # (batch_size, vocab_size)
        
        return logits
    
    def m_step_update(self, x, y):
        """Perform M-step updates on all trees in the forest"""
        # Embed tokens
        _, seq_len = x.shape
        device = x.device
        
        # Embed tokens
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding[:, :seq_len, :]
        emb = token_emb + pos_emb
        
        # Target embeddings (for regression)
        y_emb = self.token_embedding(y)
        
        # Update trees in all granularity levels
        for level, trees in self.forest.trees.items():
            for tree in trees:
                # Forward pass to get responsibilities
                _, leaf_probs = tree(emb)
                
                # M-step update
                tree.m_step_update(emb, y_emb, leaf_probs)
        
        return True
    
    def save_checkpoint(self, optimizer, step):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"{self.config.model_name}_step_{step}.pt"
        )
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path, device=None):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create config from saved dict
        config = ModelConfig()
        for key, value in checkpoint['config'].items():
            setattr(config, key, value)
        
        # Update device if specified
        if device is not None:
            config.device = device
        
        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint

# --- Training Functions ---
def train_model(model, train_dataloader, config):
    """Train the model"""
    model.to(config.device)
    model.train()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    def lr_lambda(step):
        # Linear warmup followed by cosine decay
        if step < config.warmup_steps:
            return float(step) / float(max(1, config.warmup_steps))
        else:
            decay_steps = max(1, config.max_training_steps - config.warmup_steps)
            step = min(step - config.warmup_steps, decay_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
            return cosine_decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    step = 0
    epoch = 0
    running_loss = 0.0
    start_time = time.time()
    
    while step < config.max_training_steps:
        epoch += 1
        print(f"Starting epoch {epoch}")
        
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x = x.to(config.device)
            y = y.to(config.device)
            
            # Standard gradient-based update
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # EM update (can be done less frequently)
            if step % 10 == 0:
                with torch.no_grad():
                    model.m_step_update(x, y)
            
            # Logging
            running_loss += loss.item()
            if step % 100 == 0:
                avg_loss = running_loss / 100 if step > 0 else running_loss
                elapsed = time.time() - start_time
                print(f"Step {step}/{config.max_training_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Time: {elapsed:.2f}s | "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0
                start_time = time.time()
            
            # Save checkpoint
            if step % 1000 == 0:
                model.save_checkpoint(optimizer, step)
            
            step += 1
            if step >= config.max_training_steps:
                break
    
    # Final checkpoint
    model.save_checkpoint(optimizer, step)
    
    return model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=40, top_p=0.9):
    """Generate text using the model"""
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[-model.config.context_length:]  # Truncate to context length
    
    # Convert to tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(model.config.device)
    
    # Generate tokens
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits from model
            logits = model(input_tensor)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to generated tokens
            generated_tokens.append(next_token)
            
            # Break if end of text
            if next_token == tokenizer.eos_token_id:
                break
            
            # Update input tensor
            next_input = torch.tensor([[next_token]], dtype=torch.long).to(model.config.device)
            input_tensor = torch.cat([input_tensor, next_input], dim=1)
            
            # Truncate if too long
            if input_tensor.size(1) > model.config.context_length:
                input_tensor = input_tensor[:, -model.config.context_length:]
    
    # Decode and return text
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

# --- Chatbot Example ---
class EMTreeChatbot:
    def __init__(self, model_path=None, device=None):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model if path provided, otherwise create a new one
        if model_path:
            self.model, self.checkpoint = EMTreeLLM.load_checkpoint(model_path, self.device)
        else:
            # Create a smaller config for demonstration
            config = ModelConfig(
                vocab_size=50257,  # GPT-2 vocab size
                context_length=128,
                embedding_dim=128,
                output_dim=128,
                num_trees=4,
                tree_depth=3,
                device=self.device
            )
            self.model = EMTreeLLM(config)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Chat history
        self.history = []
    
    def train(self, texts, batch_size=32, max_steps=1000):
        """Train the chatbot on a set of texts"""
        # Create dataset
        dataset = TextDataset(texts, self.tokenizer, self.model.config.context_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Update config
        self.model.config.max_training_steps = max_steps
        self.model.config.batch_size = batch_size
        
        # Train model
        self.model = train_model(self.model, dataloader, self.model.config)
        
        return self
    
    def chat(self, user_input, max_length=50):
        """Process user input and generate a response"""
        # Add user input to history
        self.history.append(f"User: {user_input}")
        
        # Prepare prompt with history
        prompt = "\n".join(self.history) + "\nBot: "
        
        # Generate response
        response = generate_text(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_length=max_length,
            temperature=0.7
        )
        
        # Add response to history
        self.history.append(f"Bot: {response}")
        
        return response
    
    def reset(self):
        """Reset chat history"""
        self.history = []

# --- Example usage ---
def example_usage():
    # Create a chatbot
    chatbot = EMTreeChatbot()
    
    # Sample training data
    training_texts = [
        "Hello! How are you today?",
        "I'm doing well, thank you for asking.",
        "What's the weather like?",
        "It's sunny and warm today.",
        "Can you help me with something?",
        "Of course, I'd be happy to help you.",
        "How does this model work?",
        "This model uses tree-based routing with expectation-maximization.",
        "Tell me a joke.",
        "Why don't scientists trust atoms? Because they make up everything!",
    ]
    
    # Train the chatbot
    print("Training chatbot...")
    chatbot.train(training_texts, batch_size=2, max_steps=100)
    
    # Chat example
    print("\nChat Example:")
    response = chatbot.chat("Hello there!")
    print(f"User: Hello there!")
    print(f"Bot: {response}")
    
    response = chatbot.chat("How does your system work?")
    print(f"User: How does your system work?")
    print(f"Bot: {response}")

if __name__ == "__main__":
    example_usage()
