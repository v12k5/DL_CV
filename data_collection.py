# data_collection.py

import os
import gymnasium as gym
import miniworld
import numpy as np
import cv2  # OpenCV for image saving
from tqdm import tqdm  # For progress bars
import argparse

def collect_data(num_episodes, env_name, frames_path, steps_per_episode=200):
    """
    Collects image frames from random exploration in a MiniWorld environment.

    Args:
        num_episodes (int): The number of episodes to run.
        env_name (str): The name of the MiniWorld environment.
        frames_path (str): The directory to save the collected frames.
        steps_per_episode (int): Max steps to run per episode.
    """
    print(f"--- Starting Data Collection ---")
    print(f"Environment: {env_name}, Episodes: {num_episodes}")

    # --- Setup ---
    # Create the MiniWorld environment
    env = gym.make(env_name, render_mode='rgb_array')

    # Create the directory for saving frames if it doesn't exist
    os.makedirs(frames_path, exist_ok=True)
    print(f"Saving frames to: {frames_path}")

    # --- Main Collection Loop ---
    frame_count = 0
    # Use tqdm for a nice progress bar over the episodes
    for episode in tqdm(range(num_episodes), desc="Collecting Episodes"):
        # Reset the environment at the start of each episode
        obs, info = env.reset()

        for step in range(steps_per_episode):
            # --- Action and Step ---
            # Sample a random action from the environment's action space
            action = env.action_space.sample()
            
            # Perform the action
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Save Frame ---
            # The observation 'obs' is an RGB image as a NumPy array.
            # OpenCV expects images in BGR format, so we need to convert the color channels.
            frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            
            # Define the filename for the image
            filename = os.path.join(frames_path, f"frame_{frame_count:07d}.jpg")
            cv2.imwrite(filename, frame_bgr)
            
            frame_count += 1

            # --- Episode End Condition ---
            # If the episode is over, break the inner loop and start a new one
            if terminated or truncated:
                break
    
    # --- Cleanup ---
    env.close()
    print(f"\n--- Data Collection Finished ---")
    print(f"Successfully collected and saved {frame_count} frames.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    # This allows you to run the script from the command line with custom settings
    parser = argparse.ArgumentParser(description="Collect data from MiniWorld environments.")
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to run for data collection.')
    parser.add_argument('--env_name', type=str, default='MiniWorld-Hallway-v0', help='Name of the MiniWorld environment.')
    parser.add_argument('--frames_path', type=str, default='data/frames', help='Directory to save the collected frames.')
    
    args = parser.parse_args()
    
    collect_data(args.num_episodes, args.env_name, args.frames_path)