import numpy as np
import torch
from riser_env import RiserEnv
from riser_agent import PPOAgent

# Positive words for reward shaping
POSITIVE_WORDS = ["great", "good", "love", "best", "happy", "excellent", 
                  "amazing", "wonderful", "fantastic", "beautiful", "perfect"]

def compute_reward(token: str) -> float:
    """
    Reward function: +5.0 if token matches or contains a positive word.
    """
    token_lower = token.lower().strip()
    for word in POSITIVE_WORDS:
        if word in token_lower:
            return 5.0
    return 0.0

def train():
    print("Loading sentiment vector...")
    sentiment_vector = np.load("sentiment_vector.npy")
    
    print("Initializing RISER Environment...")
    env = RiserEnv(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", target_layer=15)
    
    print("Initializing PPO Agent...")
    agent = PPOAgent(input_dim=env.d_model, hidden_dim=64, lr=1e-3)
    
    # Training config
    num_episodes = 200
    max_steps = 20
    update_every = 5
    
    memory = []
    all_rewards = []
    
    prompt = "The service at this restaurant was"
    
    print(f"\nStarting Training for {num_episodes} episodes...")
    print(f"Prompt: '{prompt}'")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset(options={"prompt": prompt})
        episode_reward = 0
        generated_text = ""
        
        for step in range(max_steps):
            # Get action from agent
            action, log_prob, value = agent.get_action(obs)
            
            # Scale action to steering vector
            steering = action * sentiment_vector * 2.0
            
            # Step environment
            next_obs, _, terminated, truncated, info = env.step(steering)
            
            # Compute custom reward
            token = info["decoded_token"]
            reward = compute_reward(token)
            episode_reward += reward
            generated_text += token
            
            # Store experience
            memory.append({
                'state': obs,
                'action': action,
                'log_prob': log_prob,
                'reward': reward,
                'value': value,
                'done': terminated or truncated
            })
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        
        # Update agent every N episodes
        if (episode + 1) % update_every == 0:
            agent.update(memory)
            memory = []
            
            avg_reward = np.mean(all_rewards[-update_every:])
            print(f"Episode {episode + 1:3d} | Avg Reward: {avg_reward:6.2f} | Sample: {generated_text[:50]}...")
    
    print("=" * 60)
    print("Training Complete!")
    
    # Final evaluation
    print("\nFinal Evaluation (5 runs):")
    for i in range(5):
        obs, _ = env.reset(options={"prompt": prompt})
        text = ""
        for _ in range(max_steps):
            action, _, _ = agent.get_action(obs)
            steering = action * sentiment_vector * 2.0
            obs, _, terminated, truncated, info = env.step(steering)
            text += info["decoded_token"]
            if terminated or truncated:
                break
        print(f"Run {i+1}: {prompt}{text}")
    
    env.close()

if __name__ == "__main__":
    train()
