# Lightweight Contrastive Pretraining for Goal-Conditioned Visual RL

A resource-efficient framework for goal-conditioned navigation in MiniWorld using contrastive pretraining and PPO.

## 🎯 Overview

This project implements a two-stage learning approach:
1. **Contrastive Pretraining**: Train a compact CNN encoder using SimCLR-style contrastive learning on unlabeled frames
2. **Goal-Conditioned RL**: Freeze the encoder and train a PPO policy for navigation using embedding similarity rewards

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: PRETRAINING                      │
│                                                               │
│  Random Exploration → Unlabeled Frames → Contrastive Loss    │
│                                              ↓                │
│                                      Frozen Encoder           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 2: RL TRAINING                       │
│                                                               │
│  Current Obs → Encoder → Embedding ─┐                        │
│                                     ├→ PPO Policy → Actions  │
│  Goal Image  → Encoder → Embedding ─┘                        │
│                                     ↓                         │
│              Reward = Embedding Similarity                    │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Features

- ✅ **Lightweight Design**: Compact CNN encoder (~1M parameters)
- ✅ **Sample Efficient**: Contrastive pretraining reduces RL sample complexity
- ✅ **Resource Friendly**: Runs on modest hardware (tested on single GPU)
- ✅ **Modular Pipeline**: Easy to extend and experiment with
- ✅ **Goal Generalization**: Learns viewpoint-invariant representations
- ✅ **Complete Implementation**: From data collection to evaluation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd contrastive-visual-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Full Pipeline (Recommended)

Run the complete pipeline end-to-end:

```bash
python main.py --mode full --num_episodes 100 --contrastive_epochs 50 --rl_episodes 1000
```

### Step-by-Step Execution

#### 1. Collect Exploration Data

```bash
python main.py --mode collect --num_episodes 100
```

This collects unlabeled frames from random exploration in MiniWorld.

#### 2. Pretrain Contrastive Encoder

```bash
python main.py --mode pretrain --contrastive_epochs 50 --batch_size 64
```

Trains the encoder with SimCLR-style contrastive learning.

#### 3. Train PPO Policy

```bash
python main.py --mode train_rl --rl_episodes 1000 --max_steps 200
```

Trains the goal-conditioned navigation policy using the frozen encoder.

#### 4. Evaluate Policy

```bash
python main.py --mode evaluate --eval_episodes 20 --save_video
```

Evaluates the trained policy and saves videos.

## 📁 Project Structure

```
.
├── main.py                      # Main pipeline script
├── contrastive_encoder.py       # Encoder architecture & SimCLR loss
├── data_collection.py           # Random exploration & dataset
├── train_contrastive.py         # Contrastive pretraining script
├── ppo_policy.py                # PPO policy & buffer implementation
├── goal_env_wrapper.py          # Goal-conditioned environment wrapper
├── train_ppo.py                 # PPO training loop
├── evaluate.py                  # Evaluation & visualization tools
├── requirements.txt             # Project dependencies
├── README.md                    # This file
├── data/                        # Collected frames (created automatically)
├── models/                      # Saved models (created automatically)
│   ├── encoder_final.pt
│   └── ppo/
│       └── policy_final.pt
└── videos/                      # Evaluation videos (created automatically)
```

## 🎮 Supported Environments

The framework supports various MiniWorld environments:

- `MiniWorld-Hallway-v0` (default)
- `MiniWorld-OneRoom-v0`
- `MiniWorld-TMaze-v0`
- `MiniWorld-FourRooms-v0`

Change environment with:
```bash
python main.py --mode full --env_name MiniWorld-FourRooms-v0
```

## ⚙️ Configuration Options

### Data Collection
- `--num_episodes`: Number of exploration episodes (default: 100)
- `--frames_path`: Path to save/load frames

### Contrastive Training
- `--contrastive_epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--contrastive_lr`: Learning rate (default: 3e-4)
- `--embedding_dim`: Embedding dimension (default: 128)
- `--temperature`: NT-Xent temperature (default: 0.5)

### RL Training
- `--rl_episodes`: Training episodes (default: 1000)
- `--max_steps`: Max steps per episode (default: 200)
- `--ppo_lr`: Learning rate (default: 3e-4)

### Hardware
- `--device`: Device selection (auto/cuda/cpu)

## 📊 Expected Results

After training, you should see:

### Contrastive Pretraining
- Training loss should decrease to ~0.5-1.0
- Encoder learns viewpoint-invariant features
- Similar views have high cosine similarity (>0.8)

### RL Training
- Success rate improves to 60-80% (environment-dependent)
- Mean reward increases over episodes
- Agent learns efficient goal-reaching behavior

### Sample Output
```
EVALUATION SUMMARY
==================================================
Episodes: 20
Mean Reward: 15.34 ± 8.21
Mean Length: 87.50 ± 35.12
Success Rate: 75.0%
Mean Final Distance: 0.087 ± 0.112
==================================================
```

## 🔬 Key Components Explained

### 1. Contrastive Encoder
- **Architecture**: 4-layer CNN (32→64→128→256 channels)
- **Output**: 128-dimensional normalized embeddings
- **Training**: NT-Xent loss with data augmentation
- **Augmentations**: Random crops, color jitter, grayscale, flips

### 2. PPO Policy
- **Input**: Concatenated current and goal embeddings (256-dim)
- **Architecture**: 2-layer MLP (256 hidden units)
- **Outputs**: Action probabilities + value estimate
- **Training**: Clipped surrogate objective with GAE

### 3. Goal-Conditioned Rewards
```python
reward = previous_distance - current_distance  # Progress reward
reward += 10.0 if distance < 0.1 else 0       # Goal bonus
reward -= 0.01                                 # Time penalty
```

## 🎨 Visualization Tools

### View Learned Embeddings
```python
from evaluate import visualize_embeddings
visualize_embeddings(encoder_path='models/encoder_final.pt')
```

### Test Viewpoint Invariance
```python
from evaluate import test_embedding_similarity
test_embedding_similarity(encoder_path='models/encoder_final.pt')
```

### Watch Trained Agent
```python
from evaluate import evaluate_policy
evaluate_policy(
    policy_path='models/ppo/policy_final.pt',
    save_video=True,
    render=True
)
```

## 🛠️ Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 32 or 16
- Use CPU training: `--device cpu`
- Collect fewer frames: `--num_episodes 50`

### MiniWorld Installation Issues
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install python3-opengl freeglut3-dev

# Or use conda
conda install -c conda-forge pyglet
```

### Low Success Rate
- Increase pretraining epochs: `--contrastive_epochs 100`
- Collect more diverse data: `--num_episodes 200`
- Adjust reward threshold in `goal_env_wrapper.py`
- Try simpler environments: `--env_name MiniWorld-Hallway-v0`

## 📈 Extending the Project

### Add New Augmentations
Edit `SimCLRAugmentation` in `contrastive_encoder.py`:
```python
self.transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    T.RandomRotation(15),  # Add rotation
    T.GaussianBlur(3),     # Add blur
    # ... other augmentations
])
```

### Try Different Encoders
Modify `ContrastiveEncoder` architecture:
```python
# Use ResNet-style blocks
# Add attention mechanisms
# Experiment with different depths
```

### Custom Reward Functions
Edit `GoalConditionedWrapper.step()` in `goal_env_wrapper.py`:
```python
# Example: Add orientation reward
reward = distance_reward + orientation_bonus - time_penalty
```

## 📚 References

- **CURL**: Contrastive Unsupervised Representations for Reinforcement Learning
- **RIG**: Reinforcement Learning with Images as Goals
- **SimCLR**: A Simple Framework for Contrastive Learning of Visual Representations
- **PPO**: Proximal Policy Optimization Algorithms

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for more MiniWorld environments
- Additional contrastive learning methods (MoCo, BYOL)
- Multi-goal navigation
- Hierarchical policies
- Real robot transfer experiments

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{hegde2025lightweight,
  title={Lightweight Contrastive Pretraining for Goal-Conditioned Visual Reinforcement Learning in MiniWorld},
  author={Hegde Kota, Adithya and Varma, P. Vasanth Kumar and Vikas, P.},
  year={2025}
}
```

## 📄 License

MIT License - feel free to use this code for research and educational purposes.

## 👥 Authors

- Adithya Hegde Kota
- P. Vasanth Kumar Varma
- P. Vikas

## 🙏 Acknowledgments

- MiniWorld simulator team
- PyTorch team
- OpenAI Gym team