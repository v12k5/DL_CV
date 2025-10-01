# evaluate.py

import os
import torch
import gymnasium as gym
import miniworld
import time
import argparse
import numpy as np
from tqdm import tqdm
import imageio  # For video saving

# --- Import custom components ---
from contrastive_encoder import ContrastiveEncoder
from goal_env_wrapper import GoalConditionedWrapper
from ppo_policy import ActorCritic

def evaluate_policy(args):
    """
    Evaluates a trained PPO agent, with options for rendering and video saving.
    """
    print("============================================================================================")
    print(f"--- Starting Evaluation ---")
    print(f"Policy: {args.policy_path}, Environment: {args.env_name}, Episodes: {args.eval_episodes}")
    print("============================================================================================")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # --- Environment Initialization ---
    try:
        base_env = gym.make(
            args.env_name,
            obs_width=84,
            obs_height=84,
            render_mode='rgb_array' # Always use rgb_array for capturing frames
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Please ensure you have installed MiniWorld and its dependencies correctly.")
        return

    # --- Model Loading ---
    # Load the frozen encoder
    try:
        # The GoalConditionedWrapper handles loading the encoder
        env = GoalConditionedWrapper(base_env, args.encoder_path, device)
    except FileNotFoundError:
        print(f"Error: Encoder file not found at '{args.encoder_path}'")
        print("Please ensure you have a trained encoder from Stage 1.")
        return
    except Exception as e:
        print(f"An error occurred while loading the encoder: {e}")
        return
        
    # State and action dimensions
    state_dim = args.embedding_dim * 2
    action_dim = env.action_space.n

    # Load the trained PPO policy
    try:
        policy = ActorCritic(state_dim, action_dim).to(device)
        policy.load_state_dict(torch.load(args.policy_path, map_location=device, weights_only=True))
        policy.eval() # Set policy to evaluation mode
        print(f"Successfully loaded trained policy from '{args.policy_path}'.")
    except FileNotFoundError:
        print(f"Error: Policy file not found at '{args.policy_path}'")
        print("Please ensure you have a trained PPO policy from Stage 2.")
        return
    except Exception as e:
        print(f"An error occurred while loading the policy: {e}")
        return

    # --- Evaluation Loop ---
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    frames = []

    if args.save_video:
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/evaluation_{args.env_name}_{int(time.time())}.mp4"
        print(f"Will save evaluation video to: {video_path}")

    # Use tqdm for a progress bar over the evaluation episodes
    for episode in tqdm(range(args.eval_episodes), desc="Evaluating Agent"):
        state_dict, info = env.reset()
        done = False
        total_reward = 0
        length = 0
        
        # Live rendering setup
        if args.render:
            # Use a separate env instance for rendering to avoid conflicts
            render_env = gym.make(args.env_name, obs_width=320, obs_height=240, render_mode='human')
            render_env.reset()
            render_env.unwrapped.agent.pos = env.unwrapped.agent.pos
            render_env.unwrapped.agent.dir = env.unwrapped.agent.dir


        while not done:
            # --- Get Embeddings ---
            with torch.no_grad():
                obs_embedding = env._get_embedding(state_dict['observation'])
                goal_embedding = env.goal_embedding
                state_tensor = torch.cat([obs_embedding, goal_embedding]).unsqueeze(0)

            # --- Select Action ---
            state_tensor = state_tensor.to(device)
            # FIX: Renamed 'action_tensor' to 'action' as the .act() method already returns an integer.
            action, _ = policy.act(state_tensor)

            # --- Step Environment ---
            # FIX: Removed the redundant .item() call.
            state_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            length += 1

            # --- Frame capturing ---
            if args.render:
                # Sync render env with the main env
                render_env.unwrapped.agent.pos = env.unwrapped.agent.pos
                render_env.unwrapped.agent.dir = env.unwrapped.agent.dir
                render_env.render()
                time.sleep(0.02) # Slow down rendering to be watchable

            if args.save_video:
                # Add goal image to the frame for context
                frame = state_dict['observation']
                goal_img = state_dict['goal']
                # Resize goal to be a small thumbnail
                goal_thumbnail = np.copy(goal_img)
                goal_thumbnail = cv2.resize(goal_thumbnail, (32, 32))
                # Add a border to the thumbnail
                goal_thumbnail = cv2.copyMakeBorder(goal_thumbnail, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                # Place thumbnail in the top-right corner
                frame[5:41, -41:-5, :] = goal_thumbnail
                frames.append(frame)

        # --- Log Episode Results ---
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        episode_successes.append(info.get('is_success', False))
        
        if args.render:
            render_env.close()

    # --- Save Video ---
    if args.save_video and frames:
        print(f"\nSaving video with {len(frames)} frames...")
        with imageio.get_writer(video_path, fps=30) as video:
            for frame in tqdm(frames, desc="Writing Video"):
                video.append_data(frame)
        print("Video saved successfully.")

    # --- Print Final Summary ---
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    success_rate = np.mean(episode_successes) * 100

    print("\n\n--- EVALUATION SUMMARY ---")
    print("==================================================")
    print(f"Episodes: {args.eval_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print("==================================================")
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    
    # --- Paths ---
    parser.add_argument('--policy_path', type=str, default='models/ppo/policy_MiniWorld-Hallway-v0.pt', help='Path to the trained PPO policy file.')
    parser.add_argument('--encoder_path', type=str, default='models/encoder_final.pt', help='Path to the pre-trained contrastive encoder.')
    
    # --- Environment ---
    parser.add_argument('--env_name', type=str, default='MiniWorld-Hallway-v0', help='Name of the MiniWorld environment to evaluate on.')
    
    # --- Evaluation ---
    parser.add_argument('--eval_episodes', type=int, default=20, help='Number of episodes to run for evaluation.')
    
    # --- Visualization ---
    parser.add_argument('--render', action='store_true', help='Render the environment live during evaluation.')
    parser.add_argument('--save_video', action='store_true', help='Save the evaluation run as an MP4 video.')

    # --- Model & Hardware ---
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the visual embedding (must match training).')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for evaluation.')

    args = parser.parse_args()
    
    # --- Additional Imports for Visualization ---
    # These are only imported if needed to keep the script lightweight.
    if args.save_video:
        import cv2 # OpenCV for image manipulation
        
    evaluate_policy(args)

