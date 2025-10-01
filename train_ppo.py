# train_ppo.py

import os
import torch
import torch.nn as nn
import gymnasium as gym
import miniworld
import time
import argparse
from tqdm import tqdm
from collections import deque

# --- Import all the custom components ---
from contrastive_encoder import ContrastiveEncoder
from goal_env_wrapper import GoalConditionedWrapper
from ppo_policy import ReplayBuffer, ActorCritic

class PPO:
    """
    This class encapsulates the PPO agent, including the policy,
    optimizer, and the update logic.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.buffer = ReplayBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.shared_net.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """Selects an action using the old policy (for stability during rollout)."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)
        
        # Store the state tensor and other data in the buffer
        self.buffer.states.append(state.cpu())
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action

    def update(self):
        """
        Updates the policy using the collected experiences in the buffer.
        """
        # --- 1. Calculate Rewards-to-Go ---
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # --- 2. Normalize rewards and convert to tensors ---
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # --- 3. Convert old data to tensors ---
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # --- 4. Optimize policy for K epochs ---
        for _ in range(self.K_epochs):
            # Evaluate old actions and values to get new ones
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # --- 5. Calculate the PPO surrogate loss ---
            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Advantage A(s,a)
            advantages = rewards - state_values.detach()
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of policy
            # We add a negative sign because we want to maximize this objective.
            # We add an entropy bonus to encourage exploration.
            # FIX: Use .squeeze() on state_values to match the shape of rewards and fix the warning.
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy
            
            # --- 6. Backpropagate and update weights ---
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # --- 7. Copy new weights to old policy ---
        self.policy_old.load_state_dict(self.policy.state_dict())

        # --- 8. Clear buffer for next rollout ---
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def train(args):
    """The main training function."""
    print("============================================================================================")
    print(f"--- Starting PPO Training ---")
    print(f"Environment: {args.env_name}, RL Episodes: {args.rl_episodes}")
    print("============================================================================================")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Create base environment
    base_env = gym.make(
        args.env_name,
        obs_width=84,
        obs_height=84,
        max_episode_steps=args.max_steps,
        render_mode='rgb_array'
    )
    
    # Wrap the environment for goal-conditioning
    env = GoalConditionedWrapper(base_env, args.encoder_path, device)
    
    # State and action dimensions
    # The state is the concatenated embedding of current obs and goal
    state_dim = args.embedding_dim * 2
    action_dim = env.action_space.n
    
    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, args.ppo_lr, args.ppo_lr, args.gamma, args.k_epochs, args.eps_clip, device)

    # Logging variables
    start_time = time.time()
    rewards_log = deque(maxlen=100)
    lengths_log = deque(maxlen=100)
    success_log = deque(maxlen=100)
    
    # --- Main Training Loop ---
    for episode in tqdm(range(1, args.rl_episodes + 1), desc="Training PPO"):
        state_dict, info = env.reset()
        current_ep_reward = 0

        for t in range(1, args.max_steps + 1):
            # --- 1. Get Embeddings ---
            # This is the crucial step where the visual frontend meets the policy.
            with torch.no_grad():
                obs_embedding = env._get_embedding(state_dict['observation'])
                goal_embedding = env.goal_embedding # Already computed in wrapper
                state = torch.cat([obs_embedding, goal_embedding]).unsqueeze(0)

            # --- 2. Select and Perform Action ---
            action = ppo_agent.select_action(state)
            state_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- 3. Store Experience in Buffer ---
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            current_ep_reward += reward

            # --- 4. Update Policy ---
            # If the buffer is full, trigger a policy update.
            if len(ppo_agent.buffer.states) >= args.update_timestep:
                ppo_agent.update()

            if done:
                break
        
        # --- Logging ---
        rewards_log.append(current_ep_reward)
        lengths_log.append(t)
        success_log.append(info.get('is_success', False))

        if episode % args.log_freq == 0:
            avg_reward = sum(rewards_log) / len(rewards_log)
            avg_length = sum(lengths_log) / len(lengths_log)
            avg_success = sum(success_log) / len(success_log) * 100
            
            print(f"\nEpisode {episode} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Success Rate: {avg_success:.2f}%")
            
    # --- Cleanup and Save Model ---
    env.close()
    
    print("\n--- Training Finished ---")
    os.makedirs('models/ppo', exist_ok=True)
    save_path = f'models/ppo/policy_{args.env_name}.pt'
    ppo_agent.save(save_path)
    print(f"Trained PPO policy saved to: {save_path}")
    
    end_time = time.time()
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")
    print("============================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a PPO agent for goal-conditioned visual navigation.")
    
    # --- Environment & Paths ---
    parser.add_argument('--env_name', type=str, default='MiniWorld-Hallway-v0', help='Name of the MiniWorld environment.')
    parser.add_argument('--encoder_path', type=str, default='models/encoder_final.pt', help='Path to the pre-trained contrastive encoder.')
    
    # --- RL Training Hyperparameters ---
    parser.add_argument('--rl_episodes', type=int, default=1000, help='Total number of training episodes.')
    parser.add_argument('--max_steps', type=int, default=200, help='Max steps per episode.')
    parser.add_argument('--update_timestep', type=int, default=2000, help='Number of steps to collect before updating the policy.')
    parser.add_argument('--ppo_lr', type=float, default=3e-4, help='Learning rate for PPO actor and critic.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--k_epochs', type=int, default=4, help='Number of update epochs for PPO.')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clipping parameter.')
    
    # --- Model & Hardware ---
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the visual embedding.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training.')
    
    # --- Logging ---
    parser.add_argument('--log_freq', type=int, default=20, help='Frequency of logging training progress (in episodes).')
    
    args = parser.parse_args()
    train(args)

