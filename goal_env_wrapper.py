import gymnasium as gym
import torch
import numpy as np
from contrastive_encoder import ContrastiveEncoder  # Import the encoder class

class GoalConditionedWrapper(gym.Wrapper):
    """
    Wraps a MiniWorld environment to make it goal-conditioned.
    - Goals are sampled as images from the environment.
    - Rewards are based on reduction in embedding distance (1 - cosine similarity).
    - Episode terminates if goal is reached or max steps exceeded.
    """
    def __init__(self, env, encoder_path, device='cpu', goal_threshold=0.1, max_steps=200):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The base MiniWorld environment.
            encoder_path (str): Path to the frozen pretrained encoder.
            device (str): Device for encoder ('cuda' or 'cpu').
            goal_threshold (float): Distance threshold for goal success.
            max_steps (int): Maximum steps per episode.
        """
        super().__init__(env)
        self.device = torch.device(device)
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.current_steps = 0

        # Load frozen encoder
        self.encoder = ContrastiveEncoder()
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.encoder.to(self.device)
        self.encoder.eval()  # Freeze for inference
        print(f"Loaded frozen encoder from {encoder_path} on {device}")

        # Observation space: RGB image (same as base env)
        self.observation_space = env.observation_space

        # Goal will be an RGB image
        self.goal_obs = None
        self.goal_embedding = None
        self.current_embedding = None
        self.previous_distance = None

    def reset(self, **kwargs):
        """Resets the environment, samples a new goal, and computes initial embeddings."""
        obs, info = self.env.reset(**kwargs)
        self.current_steps = 0

        # Sample goal: Step forward randomly and capture an image as goal
        for _ in range(np.random.randint(10, 50)):  # Random steps to diverse goal
            action = self.env.action_space.sample()
            goal_obs, _, _, _, _ = self.env.step(action)
        self.goal_obs = goal_obs

        # Compute embeddings
        with torch.no_grad():
            obs_tensor = self._preprocess_obs(obs).to(self.device)
            goal_tensor = self._preprocess_obs(self.goal_obs).to(self.device)
            self.current_embedding = self.encoder(obs_tensor)
            self.goal_embedding = self.encoder(goal_tensor)

        # Initial distance (1 - cosine sim)
        self.previous_distance = 1 - torch.nn.functional.cosine_similarity(
            self.current_embedding, self.goal_embedding, dim=1
        ).item()

        # Debugging: Print initial distance
        print(f"Initial distance to goal: {self.previous_distance:.4f}")

        return obs, info

    def step(self, action):
        """Steps the environment and computes reward based on embedding progress."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_steps += 1

        # Update current embedding
        with torch.no_grad():
            obs_tensor = self._preprocess_obs(obs).to(self.device)
            self.current_embedding = self.encoder(obs_tensor)

        # Current distance
        current_distance = 1 - torch.nn.functional.cosine_similarity(
            self.current_embedding, self.goal_embedding, dim=1
        ).item()

        # Reward calculation
        reward = self.previous_distance - current_distance  # Progress reward
        if current_distance < self.goal_threshold:
            reward += 10.0  # Goal bonus
            terminated = True
            info['success'] = True
        reward -= 0.01  # Time penalty

        self.previous_distance = current_distance

        # Max steps termination
        if self.current_steps >= self.max_steps:
            truncated = True

        # Debugging: Log reward components if needed
        # print(f"Step {self.current_steps}: Distance={current_distance:.4f}, Reward={reward:.4f}")

        return obs, reward, terminated, truncated, info

    def _preprocess_obs(self, obs):
        """Preprocesses RGB observation to tensor for encoder."""
        obs = np.transpose(obs, (2, 0, 1))  # HWC to CHW
        obs = torch.from_numpy(obs).float() / 255.0  # Normalize to [0,1]
        obs = obs.unsqueeze(0)  # Add batch dim
        return obs