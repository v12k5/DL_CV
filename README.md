🤖 Lightweight Contrastive Pretraining for Visual RL
A resource-efficient framework for training a goal-conditioned navigation agent in a 3D world.
This project implements a powerful two-stage learning pipeline that first teaches an agent to understand its visual world through contrastive learning, and then uses that knowledge to efficiently learn how to act and navigate to a visual goal using Reinforcement Learning.

🎯 The Core Idea: See, Then Do
Training an agent from pixels is notoriously data-hungry. This framework dramatically improves sample efficiency by decoupling visual representation learning from policy learning.

🧠 Stage 1: Learning to See (Contrastive Pretraining)
The agent's "visual cortex"—a lightweight CNN Encoder—is first pre-trained on thousands of unlabeled images collected from random exploration. Using a SimCLR-style contrastive loss, it learns to generate similar numerical representations (embeddings) for different views of the same scene and different representations for distinct scenes. This teaches it a robust, viewpoint-invariant understanding of the world.

🏆 Stage 2: Learning to Act (Reinforcement Learning)
With the visual encoder now frozen, we train a PPO policy. The agent is given a goal image and its current view. Both are passed through the frozen encoder. The policy receives these two embeddings and learns to take actions that minimize the distance between them, guided by a simple reward function. Because the agent already understands images, it can learn to navigate much faster.

🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│                 🧠 STAGE 1: PRETRAINING (Unsupervised)        │
│                                                             │
│  Random Exploration ──> Unlabeled Frames ──> Contrastive Loss │
│                                                │              │
│                                                ▼              │
│                                        ✨ Frozen Encoder      │
└─────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  🏆 STAGE 2: RL TRAINING (Goal-Conditioned)    │
│                                                             │
│  Current View ─> Encoder ─> Embedding ─┐                    │
│                                         ├─> PPO Policy ──> Action
│   Goal Image  ─> Encoder ─> Embedding ─┘                    │
│                                         │                     │
│                                         ▼                     │
│                       Reward = Embedding Similarity           │
└─────────────────────────────────────────────────────────────┘

✅ Key Features
Lightweight Design: Uses a compact CNN encoder (~1M parameters) that runs on modest hardware.

Sample Efficient: Drastically reduces the number of environment interactions needed for RL training.

Goal-Oriented: Learns to navigate to goals specified by images, not coordinates.

Modular & Clear: The code is cleanly separated into data collection, pre-training, RL training, and evaluation.

Complete Pipeline: Provides all the scripts needed to go from a blank slate to a fully trained and evaluated agent.

🚀 Quick Start Guide
1. Installation
Clone the repository, create a virtual environment, and install the required dependencies.

# Clone this repository
git clone <your-repo-url>
cd your-repo-folder

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

2. Running the Full Pipeline (Recommended)
This is the easiest way to run the entire project. The following command will automatically:

Collect 100 episodes of random exploration data.

Pre-train the contrastive encoder for 50 epochs on that data.

Train the PPO agent for 1000 episodes using the frozen encoder.

# This command runs the whole show!
python main.py --mode full

(Note: You will need to create a main.py script that orchestrates the calls to the other scripts based on the --mode flag, or simply run the individual steps below.)

3. Step-by-Step Execution
If you prefer to run each stage manually, follow these steps in order.

Step 1: Collect Exploration Data
This populates the data/frames directory with images.

python data_collection.py --num_episodes 100

Step 2: Pre-train the Contrastive Encoder
This trains the encoder on the collected frames and saves the model to models/encoder_final.pt.

python train_contrastive.py --contrastive_epochs 50

Step 3: Train the PPO Policy
This uses the frozen encoder to train the navigation agent and saves the final policy to models/ppo/.

python train_ppo.py --rl_episodes 1000

🎬 Evaluating Your Agent
Once your agent is trained, you can watch it in action!

To watch the agent navigate live:
This command will open a window and render the agent's performance for 20 episodes.

python evaluate.py --render

To save a video of the agent's performance:
This will create a videos/ directory and save an MP4 file of the evaluation run.

python evaluate.py --save_video

After running, you will see a final performance summary in your terminal:

--- EVALUATION SUMMARY ---
==================================================
Episodes: 20
Mean Reward: 0.70 ± 5.04
Mean Length: 148.95 ± 93.42
Success Rate: 20.0%
==================================================

🛠️ Extending the Project
This framework is a great starting point. Here are some ideas for extending it:

Try Harder Environments: Test how well the agent generalizes to MiniWorld-FourRooms-v0 or MiniWorld-TMaze-v0.

Improve the Encoder: Replace the simple CNN with a more powerful architecture like a ResNet.

Experiment with RL Algorithms: Implement an off-policy algorithm like SAC and compare its sample efficiency to PPO.

Enhance the Reward Function: Add penalties for colliding with walls or rewards for facing the goal to guide learning.

📚 Acknowledgments
This project is inspired by and builds upon the ideas presented in several foundational papers:

A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)

Proximal Policy Optimization Algorithms (PPO)

CURL: Contrastive Unsupervised Representations for Reinforcement Learning