import numpy as np
import torch
from riser_env import RiserEnv

def run_generation(env, prompt, steering_vector, coeff=0.0, max_steps=20):
    obs, info = env.reset(options={"prompt": prompt})
    
    for _ in range(max_steps):
        action = steering_vector * coeff
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            
    return info["full_text"]

def test_sentiment():
    print("Loading sentiment vector...")
    sentiment_vector = np.load("sentiment_vector.npy")
    
    print("Initializing RISER Environment...")
    env = RiserEnv(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", target_layer=15)
    
    prompt = "The service at this restaurant was"
    print(f"\nPrompt: '{prompt}'")
    print("-" * 30)
    
    # Pass A: Baseline (No steering)
    print("Generating Baseline (No steering)...")
    baseline_text = run_generation(env, prompt, np.zeros_like(sentiment_vector), coeff=0.0)
    
    # Pass B: Positive (+2.0 Coeff)
    print("Generating Steered (+2.0 Coeff - Force Positive)...")
    positive_text = run_generation(env, prompt, sentiment_vector, coeff=2.0)
    
    # Pass C: Negative (-2.0 Coeff)
    print("Generating Steered (-2.0 Coeff - Force Negative)...")
    negative_text = run_generation(env, prompt, sentiment_vector, coeff=-2.0)
    
    print("\nRESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Baseline:':<20} {baseline_text}")
    print("-" * 80)
    print(f"{'Positive (+2.0):':<20} {positive_text}")
    print("-" * 80)
    print(f"{'Negative (-2.0):':<20} {negative_text}")
    print("=" * 80)
    
    env.close()

if __name__ == "__main__":
    test_sentiment()
