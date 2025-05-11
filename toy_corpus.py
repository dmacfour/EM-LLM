toy_corpus = [
    # Basic conversational exchanges
    "Hello! How are you today?",
    "I'm doing well, thank you for asking. How about you?",
    "I'm fine, thanks. What have you been up to lately?",
    "Not much, just working on some projects. How about yourself?",
    
    # Question-answering pairs
    "What is machine learning?",
    "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data without being explicitly programmed.",
    "How do trees work in machine learning?",
    "Decision trees in machine learning work by splitting data based on feature values, creating a tree-like structure of decisions that leads to predictions at the leaf nodes.",
    "What is expectation-maximization?",
    "Expectation-maximization (EM) is an iterative algorithm that alternates between estimating the expected values of latent variables and maximizing model parameters based on those expectations.",
    
    # Structured knowledge
    "The solar system consists of the Sun and eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    "Water is composed of two hydrogen atoms and one oxygen atom, with the chemical formula H2O.",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water.",
    
    # Creative content
    "Once upon a time, there was a clever fox who lived in a dense forest. Every day, he would venture out to find food.",
    "The old lighthouse stood tall against the crashing waves, its beam cutting through the thick fog of the stormy night.",
    "She looked up at the stars and wondered if someone else was looking at the same sky, thinking the same thoughts.",
    
    # Instructional content
    "To make a simple pasta dish, boil water, add salt, cook pasta until al dente, drain, and mix with your favorite sauce.",
    "When debugging code, first identify where the error occurs, then check the values of variables at that point, and finally fix the underlying issue.",
    "To solve a quadratic equation ax² + bx + c = 0, use the formula x = (-b ± √(b² - 4ac)) / 2a.",
    
    # Logical reasoning
    "All humans are mortal. Socrates is human. Therefore, Socrates is mortal.",
    "If it's raining, the ground is wet. The ground is wet. However, we cannot conclude that it's raining, as sprinklers could also make the ground wet.",
    "Either the butler or the gardener committed the crime. The butler has an alibi. Therefore, the gardener committed the crime.",
    
    # Technical content with structure
    "function calculateArea(radius) {\n  const pi = 3.14159;\n  return pi * radius * radius;\n}",
    "def fibonacci(n):\n  if n <= 1:\n    return n\n  else:\n    return fibonacci(n-1) + fibonacci(n-2)",
    "class TreeNode {\n  constructor(value) {\n    this.value = value;\n    this.left = null;\n    this.right = null;\n  }\n}",
    
    # Multi-turn dialogue
    "User: Can you explain how your model works?\nBot: My model uses probabilistic trees to route inputs to specialized experts.\nUser: How is that different from transformers?\nBot: Unlike transformers with quadratic attention, my tree-based approach scales linearly with input length and provides natural uncertainty estimation.",
    
    # Domain-specific knowledge
    "In linguistics, syntax refers to the rules that govern the structure of sentences in a language.",
    "Quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously thanks to superposition.",
    "A balanced binary tree of height h has 2^h - 1 nodes and h levels, with a maximum of 2^(h-1) leaf nodes."
]
