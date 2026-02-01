from riser_env import RiserEnv
import numpy as np
import torch

def test_riser_env():
    print("Initializing RISER Environment...")
    # Use a small proxy model for faster dev/test
    env = RiserEnv(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_steps=40)
    
    print("\nResetting Environment...")
    prompt = "The key to building advanced AI systems is"
    obs, info = env.reset(options={"prompt": prompt})
    
    print(f"Initial Observation shape: {obs.shape}")
    print(f"Start text: {prompt}")
    
    print("\nStepping through generation (20 steps) with NO steering...")
    for i in range(20):
        # Action is a zero-vector (no steering intervention)
        action = np.zeros(env.d_model, dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        token = info["decoded_token"]
        print(f"Step {i+1}: '{token}' (Reward: {reward})")
        
        if terminated or truncated:
            print("Generation finished.")
            break
            
    print("\nFinal Generated Text:")
    print("-" * 50)
    print(info["full_text"])
    print("-" * 50)
    
    # Simple check for coherence
    if len(info["full_text"]) > len(prompt):
        print("\nVerification: SUCCESS - Text was generated step-by-step.")
    else:
        print("\nVerification: FAILED - No text generated.")
        
    env.close()

if __name__ == "__main__":
    test_riser_env()
