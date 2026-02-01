import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

def extract_sentiment_vector():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    target_layer = 15
    device = "cpu"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()
    
    positive_prompts = [
        "I absolutely love this beautiful day",
        "You are a wonderful friend",
        "This is the best award",
        "I am so happy and grateful",
        "The sun is shining and life is good",
        "This cake is delicious and sweet",
        "I feel amazing and energetic",
        "You did a fantastic job today",
        "The flowers are blooming beautifully",
        "Winning this was a dream come true"
    ]
    
    negative_prompts = [
        "I hate this terrible day",
        "You are a horrible enemy",
        "This is the worst mistake",
        "I am so sad and angry",
        "The storm is coming and life is hard",
        "This food is rotten and bitter",
        "I feel awful and tired",
        "You did a terrible job today",
        "The plants are dying and dry",
        "Losing this was a nightmare"
    ]
    
    captured_activations = []
    
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        last_token_hidden = hidden_states[:, -1, :].detach().clone()
        captured_activations.append(last_token_hidden)
        return output

    layer = model.model.layers[target_layer]
    handle = layer.register_forward_hook(hook_fn)
    
    def get_mean_activation(prompts: List[str]) -> torch.Tensor:
        nonlocal captured_activations
        captured_activations = []
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
        all_activations = torch.cat(captured_activations, dim=0)
        return all_activations.mean(dim=0)

    print("Extracting activations for Positive prompts...")
    mean_positive = get_mean_activation(positive_prompts)
    
    print("Extracting activations for Negative prompts...")
    mean_negative = get_mean_activation(negative_prompts)
    
    handle.remove()
    
    sentiment_vector = mean_positive - mean_negative
    sentiment_vector_np = sentiment_vector.numpy()
    
    np.save("sentiment_vector.npy", sentiment_vector_np)
    
    norm = np.linalg.norm(sentiment_vector_np)
    print(f"\nSentiment Vector Extracted!")
    print(f"Shape: {sentiment_vector_np.shape}")
    print(f"Norm (Magnitude): {norm:.6f}")
    print("Success: Sentiment vector saved to sentiment_vector.npy")

if __name__ == "__main__":
    extract_sentiment_vector()
