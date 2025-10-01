# goal_env_wrapper.py

# **Note
#---
# Checked and Working
#---

import os
import gymnasium as gym
import miniworld
import torch
import numpy as np
import torchvision.transforms as T
from collections import deque

# --- Import the pre-existing encoder architecture ---
# This reuses the code from Stage 1 without redefining it.
from contrastive_encoder import ContrastiveEncoder

class GoalConditionedWrapper(gym.Wrapper):
    """
    A Gymnasium Wrapper to create a goal-conditioned environment for visual RL.

    This wrapper modifies a standard MiniWorld environment to:
    1.  Provide a visual goal at the beginning of each episode.
    2.  Calculate rewards based on the cosine similarity between the current
        observation's embedding and the goal's embedding.
    3.  Use a pre-trained, frozen visual encoder to generate these embeddings.
    """

    def __init__(self, env, encoder_path, device, success_threshold=0.95, max_goal_steps=50):
        """
        Initializes the goal-conditioned environment wrapper.

        Args:
            env (gym.Env): The base MiniWorld environment.
            encoder_path (str): Path to the saved pre-trained encoder model state_dict.
            device (torch.device): The device (CPU or CUDA) to run the encoder on.
            success_threshold (float): Cosine similarity threshold to consider the goal reached.
            max_goal_steps (int): Maximum number of random steps to take to set a goal.
        """
        super().__init__(env)
        
        # --- Basic setup ---
        self.device = device
        self.success_threshold = success_threshold
        self.max_goal_steps = max_goal_steps
        print(f"--- Initializing GoalConditionedWrapper ---")
        print(f"Using device: {self.device}")
        print(f"Success similarity threshold: {self.success_threshold}")

        # --- Encoder Loading and Setup ---
        # Ensure the pre-trained model file exists before proceeding.
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Encoder model not found at '{encoder_path}'. "
                "Please ensure you have completed Stage 1 training."
            )
        
        # Initialize the encoder architecture and load the pre-trained weights.
        self.encoder = ContrastiveEncoder(embedding_dim=128).to(self.device)
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device, weights_only=True))
        
        # **Crucial Optimization**: Set the encoder to evaluation mode. 
        self.encoder.eval()
        print(f"Successfully loaded frozen encoder from '{encoder_path}'.")

        # --- Image Transformation ---
        # This should match the normalization used during pre-training, but without random augmentations.
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- State and Goal Management ---
        self.goal_image = None
        self.goal_embedding = None
        self.previous_distance = None

        # --- Define the new observation space ---
        self.observation_space = gym.spaces.Dict({
            'observation': self.env.observation_space,
            'goal': self.env.observation_space
        })
        
        # For reward shaping, we use a small deque to smooth the distance calculation.
        self.distance_buffer = deque(maxlen=5)

    def _get_embedding(self, obs: np.ndarray) -> torch.Tensor:
        """
        Helper function to compute the embedding for a single observation.
        """
        img_tensor = self.transform(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.encoder(img_tensor)
        return embedding.squeeze(0).cpu()

    def reset(self, **kwargs):
        """
        Resets the environment and establishes a new visual goal.
        """
        obs, info = self.env.reset(**kwargs)
        
        num_steps = np.random.randint(1, self.max_goal_steps + 1)
        goal_obs = obs
        for _ in range(num_steps):
            action = self.env.action_space.sample()
            goal_obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
                goal_obs = obs

        self.goal_image = goal_obs.copy()
        self.goal_embedding = self._get_embedding(self.goal_image)
        
        initial_embedding = self._get_embedding(obs)
        self.previous_distance = 1.0 - torch.nn.functional.cosine_similarity(initial_embedding, self.goal_embedding, dim=0).item()
        
        self.distance_buffer.clear()
        for _ in range(self.distance_buffer.maxlen):
            self.distance_buffer.append(self.previous_distance)

        return {'observation': obs, 'goal': self.goal_image}, info

    def step(self, action):
        """
        Executes an action, calculates the reward, and returns the next state.
        """
        next_obs, _, terminated, truncated, info = self.env.step(action)
        current_embedding = self._get_embedding(next_obs)
        
        current_distance = 1.0 - torch.nn.functional.cosine_similarity(current_embedding, self.goal_embedding, dim=0).item()
        self.distance_buffer.append(current_distance)
        
        avg_distance = np.mean(self.distance_buffer)
        
        reward = self.previous_distance - avg_distance
        
        if avg_distance < (1.0 - self.success_threshold):
            reward += 10.0
            terminated = True
            info['is_success'] = True
        else:
            info['is_success'] = False
            
        reward -= 0.01
        
        self.previous_distance = avg_distance
        
        obs_dict = {'observation': next_obs, 'goal': self.goal_image}
        return obs_dict, reward, terminated, truncated, info

