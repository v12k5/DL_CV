import os
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import argparse

# Corrected import: Remove .py extension
from ppo_policy import PPOPolicy, ReplayBuffer, compute_gae, Transition
from goal_env_wrapper import GoalConditionedWrapper

def train(args):
    """
    Main PPO training loop.
    """
    print("--- Starting PPO Training ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Setup environment
    try:
        base_env = gym.make(args.env_name, render_mode='rgb_array')
    except Exception as e:
        print(f"Error creating environment {args.env_name}: {e}")
        return
    
    try:
        env = GoalConditionedWrapper(base_env, encoder_path='models/encoder_final.pt', device=args.device, max_steps=args.max_steps)
    except FileNotFoundError:
        print("Error: Pretrained encoder not found at 'models/encoder_final.pt'. Ensure contrastive pretraining is complete.")
        return
    except Exception as e:
        print(f"Error initializing GoalConditionedWrapper: {e}")
        return

    # Input dim: 2 * embedding_dim (concat current + goal)
    input_dim = 2 * args.embedding_dim
    action_dim = env.action_space.n  # Assuming discrete actions in MiniWorld

    policy = PPOPolicy(input_dim, action_dim, lr=args.ppo_lr).to(device)
    buffer = ReplayBuffer()

    episode_rewards = []
    success_count = 0

    for episode in tqdm(range(args.rl_episodes), desc="Training Episodes"):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        trajectory_rewards = []
        trajectory_values = []
        trajectory_dones = []
        trajectory_states = []
        trajectory_actions = []
        trajectory_log_probs = []

        while not done:
            # Get concatenated state
            with torch.no_grad():
                current_emb = env.current_embedding.cpu().numpy().flatten()
                goal_emb = env.goal_embedding.cpu().numpy().flatten()
            state = np.concatenate([current_emb, goal_emb])

            action, log_prob = policy.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            value = policy.get_value(state)
            buffer.add(Transition(state, action, reward, None, log_prob, done))  # Next_state not used

            trajectory_rewards.append(reward)
            trajectory_values.append(value)
            trajectory_dones.append(done)
            trajectory_states.append(state)
            trajectory_actions.append(action)
            trajectory_log_probs.append(log_prob)

            total_reward += reward
            obs = next_obs

        # Post-episode: Compute GAE and update
        trajectory_values.append(0 if done else policy.get_value(state))  # Bootstrap last value
        advantages = compute_gae(trajectory_rewards, trajectory_values[:-1], trajectory_dones)

        # Normalize advantages for stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # PPO update (multiple epochs)
        states = torch.FloatTensor(trajectory_states).to(device)
        actions = torch.LongTensor(trajectory_actions).to(device)
        old_log_probs = torch.FloatTensor(trajectory_log_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + torch.FloatTensor(trajectory_values[:-1]).to(device)

        for _ in range(10):  # PPO epochs
            action_logits, values = policy(states)
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Clipped surrogate
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            critic_loss = (returns - values.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()

        buffer.clear()
        episode_rewards.append(total_reward)
        if info.get('success', False):
            success_count += 1

        # Debugging: Log every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            success_rate = (success_count / (episode + 1)) * 100
            print(f"Episode {episode + 1}: Avg Reward (last 10): {avg_reward:.2f}, Success Rate: {success_rate:.2f}%")

    # Save model
    os.makedirs('models/ppo', exist_ok=True)
    save_path = 'models/ppo/policy_final.pt'
    torch.save(policy.state_dict(), save_path)
    print(f"Trained policy saved to: {save_path}")

    env.close()
    print("--- PPO Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO policy for goal-conditioned navigation.")
    parser.add_argument('--env_name', type=str, default='MiniWorld-Hallway-v0', help='MiniWorld environment name.')
    parser.add_argument('--rl_episodes', type=int, default=1000, help='Number of RL training episodes.')
    parser.add_argument('--max_steps', type=int, default=200, help='Max steps per episode.')
    parser.add_argument('--ppo_lr', type=float, default=3e-4, help='PPO learning rate.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension (must match pretrained encoder).')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training.')

    args = parser.parse_args()
    train(args)