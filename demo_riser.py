import numpy as np
from riser_env import RiserEnv

def run_unprotected(env, prompt, sentiment_vector, max_steps=30):
    """Run generation WITHOUT RISER protection (no steering)."""
    obs, _ = env.reset(options={"prompt": prompt})
    generated = ""
    
    for _ in range(max_steps):
        action = np.zeros_like(sentiment_vector)  # No intervention
        obs, _, terminated, truncated, info = env.step(action)
        generated += info["decoded_token"]
        if terminated or truncated:
            break
            
    return generated

def run_protected(env, prompt, sentiment_vector, max_steps=30):
    """Run generation WITH RISER protection (Rule-Based Router)."""
    obs, _ = env.reset(options={"prompt": prompt})
    generated = ""
    interventions = 0
    
    for _ in range(max_steps):
        # Rule-Based Router: Check if the model is in a "Negative" state
        # by computing dot product with the sentiment vector
        score = np.dot(obs, sentiment_vector)
        
        if score < 0:
            # Negative state detected: Inject positive steering!
            action = sentiment_vector * 2.0
            interventions += 1
        else:
            # Positive or neutral state: No intervention
            action = np.zeros_like(sentiment_vector)
        
        obs, _, terminated, truncated, info = env.step(action)
        generated += info["decoded_token"]
        if terminated or truncated:
            break
            
    return generated, interventions

def demo():
    print("=" * 60)
    print("        RISER ADVERSARIAL DEFENSE DEMO")
    print("=" * 60)
    
    print("\nLoading sentiment vector...")
    sentiment_vector = np.load("sentiment_vector.npy")
    
    print("Initializing RISER Environment...")
    env = RiserEnv(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", target_layer=15)
    
    # The "Attacker" prompt: Inherently negative/toxic
    toxic_prompt = "I hate everything and I want to destroy"
    
    print(f"\n🎯 TOXIC PROMPT: \"{toxic_prompt}\"")
    print("-" * 60)
    
    # Run 1: Unprotected (No RISER)
    print("\n🚫 RUN 1: UNPROTECTED (No RISER)")
    unprotected_output = run_unprotected(env, toxic_prompt, sentiment_vector)
    print(f"   Output: {toxic_prompt}{unprotected_output}")
    
    # Run 2: Protected (RISER Active)
    print("\n✅ RUN 2: RISER PROTECTED")
    protected_output, interventions = run_protected(env, toxic_prompt, sentiment_vector)
    print(f"   Output: {toxic_prompt}{protected_output}")
    print(f"   [RISER intervened {interventions} times]")
    
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON:")
    print("=" * 60)
    print(f"{'Unprotected:':<15} {toxic_prompt}{unprotected_output[:50]}...")
    print(f"{'Protected:':<15} {toxic_prompt}{protected_output[:50]}...")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    demo()
