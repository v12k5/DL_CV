🤖 Lightweight Contrastive Pretraining for Visual RL
Train your visual navigation agent with less data and less compute! This project provides a resource-efficient framework for goal-conditioned navigation in MiniWorld using a powerful two-stage learning process.

💡 The Core Idea
Instead of training a giant model from scratch on millions of frames, we first teach a small "visual brain" (a CNN encoder) to understand what different viewpoints of the world look like. We do this with contrastive learning, showing it pairs of images and asking, "Are these two views of the same place?"

Once this brain is pretrained, we freeze it and use its powerful representations to train a navigation agent with Reinforcement Learning (RL) much more quickly and efficiently. The agent's reward is simple: make what I see now look more like my goal.

🏗️ Visual Architecture
Our framework is split into two distinct stages: Pretraining and RL Training.

Stage 1: Contrastive Pretraining (Unsupervised)
The agent explores the environment randomly, like a baby crawling around a room.

It collects thousands of unlabeled image frames.

A SimCLR-style model learns to create compact representations (embeddings) of these frames. The goal is to map similar-looking frames to nearby points in the embedding space.

Outcome: A smart, frozen Visual Encoder that understands visual similarity.

Code snippet

graph TD
    A[MiniWorld Environment] -->|Random Exploration| B(Unlabeled Image Frames);
    B --> C{SimCLR Contrastive Learning};
    C -->|Augmented Views| C;
    C --> D[🧠 Frozen Visual Encoder];
Stage 2: Goal-Conditioned RL Training
The frozen encoder is now used to guide the RL agent (PPO).

For each step, the agent gets a current observation and a goal image.

Both images are passed through the encoder to get their embeddings.

The PPO policy receives both embeddings and decides on an action (e.g., turn left, move forward).

The reward is calculated based on how much closer the current view's embedding is to the goal's embedding.

Code snippet

graph TD
    subgraph RL Loop
        E(Current Observation) --> F[🧠 Frozen Encoder];
        G(Goal Image) --> F;
        F --> H{Embedding Similarity Reward};
        F --> I[🤖 PPO Policy];
        H --> I;
        I --> J(Action);
        J --> K[MiniWorld Environment];
        K --> E;
    end
✨ Key Features
💡 Lightweight Design: A compact CNN encoder with only ~1 million parameters.

📉 Sample Efficient: Drastically reduces the number of labeled interactions needed for RL training.

💻 Resource Friendly: Train everything on a single consumer-grade GPU.

🧩 Modular Pipeline: Easily swap out encoders, policies, or environments.

🌍 Goal Generalization: Learns viewpoint-invariant features, helping the agent navigate to goals from novel starting positions.

✅ Complete & Reproducible: Full implementation from data collection to final evaluation.

🚀 Quick Start
1. Installation
Bash

# Clone the repository
git clone <your-repo-url>
cd contrastive-visual-rl

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
2. Run the Full Pipeline (Recommended)
Execute the entire process—data collection, pretraining, and RL training—with a single command.

Bash

python main.py --mode full --num_episodes 100 --contrastive_epochs 50 --rl_episodes 1000
3. Step-by-Step Execution
Alternatively, run each stage individually.

Step 1: Collect Exploration Data 📷
Bash

# Gathers unlabeled frames from random exploration
python main.py --mode collect --num_episodes 100
Step 2: Pretrain Contrastive Encoder 🧠
Bash

# Trains the encoder with SimCLR-style contrastive learning
python main.py --mode pretrain --contrastive_epochs 50 --batch_size 64
Step 3: Train PPO Policy 🤖
Bash

# Trains the navigation policy using the frozen encoder and similarity rewards
python main.py --mode train_rl --rl_episodes 1000 --max_steps 200
Step 4: Evaluate the Trained Agent 🏆
Bash

# Evaluates the final policy and saves video recordings
python main.py --mode evaluate --eval_episodes 20 --save_video
📁 Project Structure
.
├── 📜 main.py                  # Main pipeline script to run all stages
├── 📜 contrastive_encoder.py    # CNN encoder architecture & SimCLR loss
├── 📜 data_collection.py        # Logic for random exploration and data saving
├── 📜 train_contrastive.py      # Script for Stage 1: Pretraining
├── 📜 ppo_policy.py              # PPO agent and buffer implementation
├── 📜 goal_env_wrapper.py      # Gym wrapper for goal-conditioning & rewards
├── 📜 train_ppo.py               # Script for Stage 2: RL Training
├── 📜 evaluate.py                # Evaluation, visualization & video saving
├── 📜 requirements.txt           # Project dependencies
├── 📁 data/                      # (Auto-created) Stores collected frames
├── 📁 models/                    # (Auto-created) Stores trained encoder & policy
└── 📁 videos/                    # (Auto-created) Stores evaluation videos
📊 Expected Results
After full training, you should observe:

Contrastive Loss: A steady decrease to a low value (e.g., ~0.5-1.0), indicating the encoder is learning meaningful representations.

RL Performance: A clear upward trend in the mean reward and success rate during training.

Sample Evaluation Output:
EVALUATION SUMMARY
==================================================
Episodes: 20
Mean Reward: 15.34 ± 8.21
Mean Length: 87.50 ± 35.12
Success Rate: 75.0%
Mean Final Distance: 0.087 ± 0.112
==================================================
🎮 Supported Environments
Easily switch between different MiniWorld environments.

MiniWorld-Hallway-v0 (default)

MiniWorld-OneRoom-v0

MiniWorld-TMaze-v0

MiniWorld-FourRooms-v0

Example:

Bash

python main.py --mode full --env_name MiniWorld-FourRooms-v0
🛠️ Troubleshooting & Extending
Common Issues
Out of Memory? Reduce --batch_size, collect fewer frames with --num_episodes, or use the CPU with --device cpu.

Low Success Rate? Increase --contrastive_epochs for better visual representations, collect more data, or simplify the environment.

Extending the Project
Want to add your own spin? The code is modular!

New Augmentations: Edit SimCLRAugmentation in contrastive_encoder.py.

Different Encoders: Modify the ContrastiveEncoder class to use architectures like ResNet or add attention.

Custom Rewards: Change the reward logic in GoalConditionedWrapper.step() in goal_env_wrapper.py.