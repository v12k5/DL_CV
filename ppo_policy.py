import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple

# Define a transition tuple for the buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'log_prob', 'done'))

class PPOPolicy(nn.Module):
    """
    PPO Actor-Critic Policy Network.
    - Actor: Outputs action probabilities.
    - Critic: Outputs value estimates.
    """
    def __init__(self, input_dim, action_dim, hidden_dim=256, lr=3e-4):
        """
        Initializes the policy.

        Args:
            input_dim (int): Dimension of input (concatenated embeddings).
            action_dim (int): Number of actions in the environment.
            hidden_dim (int): Hidden layer size.
            lr (float): Learning rate for optimizer.
        """
        super().__init__()
        # Shared base
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.action_dim = action_dim

    def forward(self, state):
        """Forward pass: Returns action logits and value."""
        features = self.base(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def select_action(self, state):
        """Samples action and returns action + log_prob."""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def get_value(self, state):
        """Returns value estimate for state."""
        state = torch.FloatTensor(state).unsqueeze(0)
        _, value = self.forward(state)
        return value.item()

class ReplayBuffer:
    """
    Buffer to store trajectories for PPO updates.
    """
    def __init__(self):
        self.buffer = []

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self):
        return self.buffer

    def clear(self):
        self.buffer = []

def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    Computes Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae = 0
    next_value = 0 if dones[-1] else values[-1]  # Bootstrap if not done
    for reward, value, done in reversed(list(zip(rewards, values, dones))):
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambda_ * (1 - done) * gae
        advantages.insert(0, gae)
        next_value = value
    return advantages