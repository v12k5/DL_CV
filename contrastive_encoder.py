# contrastive_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class SimCLRAugmentation:
    """
    Applies SimCLR-style data augmentations to a single image to create a positive pair.
    """
    def __init__(self, img_size=84):
        # The series of transformations to apply
        self.transform = T.Compose([
            # THIS IS THE FIX: The T.ToPILImage() line that was here has been removed.
            # The input from the Dataset is already a PIL Image.
            T.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(), # Convert the PIL Image to a tensor
            # Normalize the tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        """
        Takes a PIL image and returns two randomly augmented versions of it as tensors.
        """
        return self.transform(x), self.transform(x)

class ContrastiveEncoder(nn.Module):
    """
    The lightweight CNN encoder and projector head.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        
        # --- CNN Base (Feature Extractor) ---
        self.cnn_base = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # --- Projector Head ---
        self.projector = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.cnn_base(x)
        embedding = self.projector(features)
        return F.normalize(embedding, dim=1)