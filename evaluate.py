import os
import gymnasium as gym
import torch
import numpy as np
import cv2  # For video saving
from tqdm import tqdm
import argparse

from goal_env_wrapper import GoalConditionedWrapper
from ppo_policy import PPOPolicy

def evaluate(args):
    """
    Evaluates the trained PPO policy.
    """
    print("--- Starting Evaluation ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # Setup environment
    base_env = gym.make(args.env_name, render_mode='rgb_array')
    env = GoalConditionedWrapper(base_env, encoder_path='models/encoder_final.pt', device=args.device, max_steps=args.max_steps)

    # Load policy
    input_dim = 2 * args.embedding_dim
    action_dim = env.action_space.n
    policy = PPOPolicy(input_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(args.policy_path, map_location=device))
    policy.eval()
    print(f"Loaded policy from {args.policy_path}")

    rewards = []
    lengths = []
    successes = []
    final_distances = []

    for episode in tqdm(range(args.eval_episodes), desc="Evaluation Episodes"):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        frames = [] if args.save_video else None

        while not done:
            # Get state
            with torch.no_grad():
                current_emb = env.current_embedding.cpu().numpy().flatten()
                goal_emb = env.goal_embedding.cpu().numpy().flatten()
            state = np.concatenate([current_emb, goal_emb])

            action, _ = policy.select_action(state)  # Deterministic? Or sample? Here sample for variety
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if args.save_video:
                frames.append(obs)

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(info.get('success', False))
        final_distances.append(env.previous_distance)  # Last distance

        # Save video if requested
        if args.save_video:
            os.makedirs('videos', exist_ok=True)
            video_path = f'videos/eval_episode_{episode}.mp4'
            height, width, _ = frames[0].shape
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"Saved video: {video_path}")

    # Summary statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    success_rate = (np.sum(successes) / args.eval_episodes) * 100
    mean_final_dist = np.mean(final_distances)
    std_final_dist = np.std(final_distances)

    print("\nEVALUATION SUMMARY")
    print("==================================================")
    print(f"Episodes: {args.eval_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Mean Final Distance: {mean_final_dist:.3f} ± {std_final_dist:.3f}")
    print("==================================================")

    env.close()
    print("--- Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy.")
    parser.add_argument('--env_name', type=str, default='MiniWorld-Hallway-v0', help='MiniWorld environment name.')
    parser.add_argument('--eval_episodes', type=int, default=20, help='Number of evaluation episodes.')
    parser.add_argument('--max_steps', type=int, default=200, help='Max steps per episode.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--policy_path', type=str, default='models/ppo/policy_final.pt', help='Path to trained policy.')
    parser.add_argument('--save_video', action='store_true', help='Save evaluation videos.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device for evaluation.')

    args = parser.parse_args()
    evaluate(args)