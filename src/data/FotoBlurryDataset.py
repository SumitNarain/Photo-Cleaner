import glob
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224

data_transforms = transforms.Compose(
    [
        # 1. Standardizes size for model input
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # 2. Converts to Tensor (HWC -> CWH) and scales values to [0.0, 1.0]
        transforms.ToTensor(),
        # 3. Final normalization using ImageNet statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class FotoBlurryDataset(Dataset):
    """
    Dataset class must have 3 functions implemented: __init__(), __len__(), __getitem__()
    Initialize the dataset, Return number of samples, and return a single sample given an index, respectively

    We need to assign whether the photo is blurry or not so encode blurry (1) and sharp (0)
    """

    def __init__(self, data_path, transform=None):
        # data loading
        self.transform = transform
        self.all_image_paths = []
        self.all_labels = []

        # Blurred Dataset and Motion Blurred
        blurred_dirs = ["defocused_blurred", "motion_blurred"]
        for blur_type in blurred_dirs:
            blur_dir = os.path.join(data_path, blur_type)
            blur_paths = glob.glob(os.path.join(blur_dir, "*.jpg"))
            self.all_image_paths.extend(blur_paths)
            self.all_labels.extend([1] * len(blur_paths))

        # Sharp Dataset
        sharp_directory = os.path.join(data_path, "sharp")
        list_of_sharp_paths = glob.glob(os.path.join(sharp_directory, "*.jpg"))
        self.all_image_paths.extend(list_of_sharp_paths)
        self.all_labels.extend([0] * len(list_of_sharp_paths))

        print(f"Total images found: {len(self.all_image_paths)}")
        print(f"Sharp images (0): {self.all_labels.count(0)}")
        print(f"Blurred images (1): {self.all_labels.count(1)}")

    def __len__(self):
        # Get all the images in the dataset; len(dataset)
        return len(self.all_image_paths)

    def __getitem__(self, index):
        # Get an individual item path and label; dataset[0]
        image_path = self.all_image_paths[index]
        image_label = self.all_labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(image_label)
