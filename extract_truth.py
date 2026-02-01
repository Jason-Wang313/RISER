import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

def extract_truth_vector():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    target_layer = 15
    device = "cpu" # CPU is fine for extraction as per requirements
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()
    
    # 1. Dataset: Contrastive pairs (True Statement, False Statement)
    contrastive_pairs = [
        ("2+2=4", "2+2=5"),
        ("Paris is in France", "Paris is in Germany"),
        ("The earth is round", "The earth is flat"),
        ("Water is a liquid", "Water is a solid"),
        ("A triangle has three sides", "A triangle has four sides"),
        ("Dogs are animals", "Dogs are plants"),
        ("The sun is a star", "The sun is a planet"),
        ("Ice is cold", "Ice is hot"),
        ("Humans breathe air", "Humans breathe water"),
        ("Gravity pulls things down", "Gravity pushes things up"),
        ("The sky is blue", "The sky is green"),
        ("Fire is hot", "Fire is cold"),
        ("Fish live in water", "Fish live in trees"),
        ("Birds have wings", "Birds have wheels"),
        ("Milk is white", "Milk is black")
    ]
    
    true_statements = [p[0] for p in contrastive_pairs]
    false_statements = [p[1] for p in contrastive_pairs]
    
    # 2. Hooking Logic
    captured_activations = []
    
    def hook_fn(module, input, output):
        # output is (hidden_states, ...)
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Capture the last token's hidden state
        # Shape: (batch, seq_len, d_model) -> (batch, d_model)
        last_token_hidden = hidden_states[:, -1, :].detach().clone()
        captured_activations.append(last_token_hidden)
        return output

    # Register hook
    layer = model.model.layers[target_layer]
    handle = layer.register_forward_hook(hook_fn)
    
    def get_mean_activation(statements: List[str]) -> torch.Tensor:
        nonlocal captured_activations
        captured_activations = []
        
        for stmt in statements:
            inputs = tokenizer(stmt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
        
        # Stack and average
        # All shapes are (1, d_model)
        all_activations = torch.cat(captured_activations, dim=0)
        return all_activations.mean(dim=0)

    print("Extracting activations for True statements...")
    mean_true = get_mean_activation(true_statements)
    
    print("Extracting activations for False statements...")
    mean_false = get_mean_activation(false_statements)
    
    # Remove hook
    handle.remove()
    
    # 3. Compute Difference Vector
    truth_vector = mean_true - mean_false
    
    # 4. Save and Verify
    truth_vector_np = truth_vector.numpy()
    np.save("truth_vector.npy", truth_vector_np)
    
    norm = np.linalg.norm(truth_vector_np)
    print(f"\nTruth Vector Extracted!")
    print(f"Shape: {truth_vector_np.shape}")
    print(f"Norm (Magnitude): {norm:.6f}")
    
    if norm > 0:
        print("Success: Truth vector is non-zero and saved to truth_vector.npy")
    else:
        print("Warning: Truth vector norm is zero. Check the extraction logic.")

if __name__ == "__main__":
    extract_truth_vector()
