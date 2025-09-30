# train_contrastive.py

import os
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import argparse

# Import the model and augmentation pipeline from the other file
from contrastive_encoder import ContrastiveEncoder, SimCLRAugmentation

class FrameDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the collected frames.
    """
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        # Find all .jpg files in the directory
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        if not self.image_paths:
            raise ValueError(f"No .jpg frames found in {root_dir}. Did you run data_collection.py?")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image using PIL
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Apply the SimCLR augmentations to get a positive pair
        view1, view2 = self.transform(image)
        return view1, view2


def nt_xent_loss(z_i, z_j, temperature):
    """
    Calculates the NT-Xent loss for contrastive learning.
    """
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_ij = torch.diag(sim_matrix, batch_size)
    sim_ji = torch.diag(sim_matrix, -batch_size)
    positive_pairs = torch.cat([sim_ij, sim_ji], dim=0)
    numerator = torch.exp(positive_pairs / temperature)
    denominator = torch.sum(torch.exp(sim_matrix / temperature) * ~mask, dim=1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


def train(args):
    """Main training function."""
    print("--- Starting Contrastive Pre-training ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    augmentations = SimCLRAugmentation()
    dataset = FrameDataset(root_dir=args.frames_path, transform=augmentations)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Found {len(dataset)} frames.")

    model = ContrastiveEncoder(embedding_dim=args.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.contrastive_lr)

    for epoch in range(args.contrastive_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.contrastive_epochs}")
        
        for view1, view2 in progress_bar:
            view1, view2 = view1.to(device), view2.to(device)
            
            z_i = model(view1)
            z_j = model(view2)
            
            loss = nt_xent_loss(z_i, z_j, args.temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.contrastive_epochs}, Average Loss: {epoch_loss:.4f}")

    print("\n--- Training Finished ---")
    os.makedirs('models', exist_ok=True)
    save_path = 'models/encoder_final.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Trained encoder saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contrastive encoder on MiniWorld frames.")
    parser.add_argument('--frames_path', type=str, default='data/frames', help='Directory where collected frames are stored.')
    parser.add_argument('--contrastive_epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--contrastive_lr', type=float, default=3e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the output embedding.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for the NT-Xent loss.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training.')
    
    args = parser.parse_args()
    train(args)