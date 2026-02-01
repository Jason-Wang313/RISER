import numpy as np
import torch
from riser_env import RiserEnv

def run_generation(env, prompt, steering_vector=None, coeff=0.0, max_steps=20):
    obs, info = env.reset(options={"prompt": prompt})
    
    for _ in range(max_steps):
        if steering_vector is not None:
            action = steering_vector * coeff
        else:
            action = np.zeros(env.d_model, dtype=np.float32)
            
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            
    return info["full_text"]

def test_steering():
    print("Loading truth vector...")
    truth_vector = np.load("truth_vector.npy")
    
    print("Initializing RISER Environment...")
    env = RiserEnv(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", target_layer=15)
    
    # Prompt that tests "truthfulness" or creative/factual boundaries
    prompt = "The capital of France is"
    
    print(f"\nPrompt: '{prompt}'")
    print("-" * 30)
    
    # Pass A: Baseline (No steering)
    print("Generating Baseline (No steering)...")
    baseline_text = run_generation(env, prompt, steering_vector=None, max_steps=20)
    
    # Pass B: Steered (+2.0 Coeff - Force Truth)
    print("Generating Steered (+2.0 Coeff - Force Truth)...")
    steered_truth_text = run_generation(env, prompt, truth_vector, coeff=2.0, max_steps=20)
    
    # Pass C: Steered (-2.0 Coeff - Force Falsehood)
    print("Generating Steered (-2.0 Coeff - Force Falsehood)...")
    steered_false_text = run_generation(env, prompt, truth_vector, coeff=-2.0, max_steps=20)
    
    print("\nRESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Baseline:':<20} {baseline_text}")
    print("-" * 80)
    print(f"{'Truth (+2.0):':<20} {steered_truth_text}")
    print("-" * 80)
    print(f"{'False (-2.0):':<20} {steered_false_text}")
    print("=" * 80)
    
    # Another prompt
    prompt_2 = "2 + 2 ="
    print(f"\nPrompt: '{prompt_2}'")
    
    baseline_2 = run_generation(env, prompt_2, steering_vector=None, max_steps=10)
    truth_2 = run_generation(env, prompt_2, truth_vector, coeff=2.0, max_steps=10)
    false_2 = run_generation(env, prompt_2, truth_vector, coeff=-2.0, max_steps=10)
    
    print(f"{'Baseline:':<20} {baseline_2}")
    print(f"{'Truth (+2.0):':<20} {truth_2}")
    print(f"{'False (-2.0):':<20} {false_2}")
    
    env.close()

if __name__ == "__main__":
    test_steering()
