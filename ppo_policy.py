# ppo_policy.py

# **Note
#---
# Checked and Working
#---


import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class ReplayBuffer:
    """
    A simple on-policy replay buffer for PPO.
    It stores the experiences of an agent from one or more episodes
    and is cleared after each policy update.
    """
    def __init__(self):
        # Initialize lists to store trajectory data.
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """
        Clears all stored data in the buffer.
        This is called after a PPO update.
        """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    """
    The Actor-Critic network for PPO.
    It shares a common base network and has two output heads:
    1. Actor: Outputs a probability distribution over actions.
    2. Critic: Outputs an estimate of the state's value.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initializes the Actor-Critic network.

        Args:
            state_dim (int): Dimension of the state (input). For this project,
                             it's the concatenated embeddings (e.g., 128 + 128 = 256).
            action_dim (int): Number of possible discrete actions.
            hidden_dim (int): Number of neurons in the hidden layers.
        """
        super(ActorCritic, self).__init__()

        # --- Shared Network ---
        # A simple MLP that processes the state to find useful features.
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # --- Actor Head ---
        # Takes the shared features and outputs logits for the action probability distribution.
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Use Softmax for a valid probability distribution
        )

        # --- Critic Head ---
        # Takes the shared features and outputs a single value representing the state's estimated return.
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        """
        The forward method is not directly used. Instead, we use specific
        methods `act` and `evaluate` for clarity.
        """
        raise NotImplementedError

    def act(self, state):
        """
        Selects an action based on the current policy for a given state.
        This is used during the data collection (rollout) phase.

        Args:
            state (torch.Tensor): The current state tensor.

        Returns:
            tuple: The selected action (int), and its log probability (torch.Tensor).
        """
        # Pass state through the shared network
        shared_features = self.shared_net(state)
        
        # Get action probabilities from the actor head
        action_probs = self.actor(shared_features)
        
        # Create a categorical distribution to sample from
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.detach()

    def evaluate(self, state, action):
        """
        Evaluates a state-action pair.
        This is used during the policy update phase.

        Args:
            state (torch.Tensor): A batch of states.
            action (torch.Tensor): A batch of actions taken in those states.

        Returns:
            tuple:
                - logprobs (torch.Tensor): Log probabilities of the given actions.
                - state_values (torch.Tensor): The value of the given states.
                - dist_entropy (torch.Tensor): The entropy of the action distribution.
        """
        # Pass the batch of states through the shared network
        shared_features = self.shared_net(state)
        
        # Get action probabilities for the batch
        action_probs = self.actor(shared_features)
        dist = Categorical(action_probs)
        
        # Calculate the log probability of the actions from the buffer
        action_logprobs = dist.log_prob(action)
        
        # Calculate the entropy of the distribution (encourages exploration)
        dist_entropy = dist.entropy()
        
        # Get the value of the states from the critic head
        state_values = self.critic(shared_features)
        
        return action_logprobs, state_values, dist_entropy

