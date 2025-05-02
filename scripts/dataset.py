# dataset.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.unet import UNet
from torch.utils.data import Dataset
from PIL import Image

class RoadDataset(Dataset):
    def __init__(self, image_dir, target_dir, transform=None):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        target_path = os.path.join(self.target_dir, self.image_filenames[idx].replace("image", "target"))

        image = Image.open(image_path).convert("L")
        target = Image.open(target_path).convert("L")

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target