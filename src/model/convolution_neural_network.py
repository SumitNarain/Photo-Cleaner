import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input channels is 3 (RGB)
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Calculate the size of the tensor after the convolutional/pooling layers
        # For 224x224 input, it becomes 224/2/2 = 56. The channel count is 32.
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)  # num_classes = 2 (sharp or blurry)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x