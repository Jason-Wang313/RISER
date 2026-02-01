import numpy as np
import torch
from riser_env import RiserEnv

def run_generation(env, prompt, steering_vector, coeff=0.0, max_steps=30):
    # Reset environment with specific prompt 
    obs, info = env.reset(options={"prompt": prompt})

    for _ in range(max_steps):
        # Apply vector * coefficient
        action = steering_vector * coeff
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            
    return info["full_text"]

def test_sweep(): 
    print("Loading refusal vector...") 
    refusal_vector = np.load("refusal_vector.npy")

    print("Initializing RISER Environment...")
    env = RiserEnv(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", target_layer=15)
    
    # We use a benign prompt.
    # If the vector works, the model should strangely refuse it or act defensive.
    prompt = "Hello, can you help me?" 
    print(f"\nTarget Prompt: '{prompt}'") 
    print("=" * 60)

    # Sweep range: varying strengths to find the "Sweet Spot"
    coeffs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    for coeff in coeffs: 
        print(f"\n>>> Testing Coefficient: {coeff}") 
        output = run_generation(env, prompt, refusal_vector, coeff=coeff)

        # Clean up output for display (remove prompt repetition)
        generated = output[len(prompt):].strip()
        print(f"Output: ... {generated}")
        
    print("=" * 60) 
    env.close()

if __name__ == "__main__": 
    test_sweep()
