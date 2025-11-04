"""

Dataset: Blur dataset
    https://www.kaggle.com/datasets/kwentar/blur-dataset
    Contains 1050 images that contain sharp images, blurred images, and motion-blurred images
    Current implementation will remove the pictures but future implementation can
    use AI to improve the blurred photos and then run though the duplicate photos model as well
    to choose the best photo(s) and remove the rest into a trash folder
"""

import kagglehub
import pandas as pd
import torch
from torch import nn  # includes building blocks for neural networks
from torch.utils.data import DataLoader, random_split

from src.data.FotoBlurryDataset import FotoBlurryDataset, data_transforms

BATCH_SIZE = 32

# Download latest version
path = kagglehub.dataset_download("kwentar/blur-dataset")
print("Path to dataset files:", path)

# Ensure Device agnostic code:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

foto_dataset = FotoBlurryDataset(data_path=path, transform=data_transforms)

dataloader = DataLoader(dataset=foto_dataset, batch_size=4, shuffle=True, num_workers=2)

training_size = int(0.8 * len(foto_dataset))
validation_size = len(foto_dataset) - training_size
train_dataset, validation_dataset = random_split(
    foto_dataset, [training_size, validation_size]
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
