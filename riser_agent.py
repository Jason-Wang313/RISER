import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List

class PPOAgent:
    """
    Minimal PPO Agent for RISER.
    The agent learns to output a scalar alpha in [-1, 1] which scales the steering vector.
    """
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 64, lr: float = 1e-3):
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.device = torch.device("cpu")
        
        # Actor Network: Outputs mean of action distribution
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output in [-1, 1]
        ).to(self.device)
        
        # Critic Network: Outputs value estimate
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Log std for action distribution (learnable)
        self.log_std = nn.Parameter(torch.zeros(1))
        
        # Optimizers
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def get_action(self, state: np.ndarray) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Given a state, return action, log_prob, and value estimate.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Actor: Get action mean
        action_mean = self.actor(state_tensor)
        std = torch.exp(self.log_std).clamp(min=0.01, max=1.0)
        
        # Sample action from Normal distribution
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Critic: Get value
        value = self.critic(state_tensor)
        
        # Clamp action to [-1, 1]
        action_clamped = torch.clamp(action, -1.0, 1.0)
        
        return action_clamped.item(), log_prob, value
    
    def update(self, memory: List[dict]):
        """
        Perform PPO update using collected experiences.
        memory: List of dicts with keys: state, action, log_prob, reward, value, done
        """
        if len(memory) == 0:
            return
        
        # Prepare batches
        states = torch.FloatTensor(np.array([m['state'] for m in memory])).to(self.device)
        actions = torch.FloatTensor([m['action'] for m in memory]).unsqueeze(1).to(self.device)
        old_log_probs = torch.cat([m['log_prob'] for m in memory]).to(self.device)
        rewards = [m['reward'] for m in memory]
        dones = [m['done'] for m in memory]
        old_values = torch.cat([m['value'] for m in memory]).to(self.device)
        
        # Compute Returns and Advantages (GAE simplified)
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - int(d))
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = returns - old_values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update (single epoch for simplicity)
        action_mean = self.actor(states)
        std = torch.exp(self.log_std).clamp(min=0.01, max=1.0)
        dist = Normal(action_mean, std)
        new_log_probs = dist.log_prob(actions)
        
        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs.unsqueeze(1))
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        new_values = self.critic(states)
        critic_loss = nn.MSELoss()(new_values, returns)
        
        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
