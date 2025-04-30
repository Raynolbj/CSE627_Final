import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")  # Adds project root to Python path
from models.unet import UNet
from scripts.dataset import RoadDataset
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.morphology import skeletonize, closing, footprint_rectangle

from models.unet import UNet
from models.node_metrics import (
    compute_node_metrics,
    find_valence_points,
    extract_points
)


# === DATASET ===
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


# === GENTLE CLEANING FUNCTION ===
def clean_and_skeletonize(pred):
    pred = pred / pred.max()
    binary_pred = pred > 0.01
    footprint = footprint_rectangle((3, 3))
    binary_pred = closing(binary_pred, footprint)
    pred_skeleton = skeletonize(binary_pred)
    return binary_pred, pred_skeleton


# === DEBUG VALENCE INFO ===
def debug_valence_map(skeleton, label="Prediction"):
    valence_map = find_valence_points(skeleton)
    plt.figure(figsize=(6, 5))
    plt.imshow(valence_map, cmap="magma")
    plt.title(f"Valence Map of {label}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    for val in [1, 2, 3, 4]:
        pts = extract_points(valence_map, val)
        print(f"[{label}] Valence {val}: {len(pts)} points")


# === TOY TEST FUNCTION ===
def toy_junction_test():
    toy = np.zeros((10, 10), dtype=np.uint8)
    toy[5, 2:8] = 1
    toy[2:9, 5] = 1

    metrics = compute_node_metrics(toy, toy, max_dist=3)
    print("\n✅ Toy Junction Metric Result (should show valence-4 TP=1):")
    for val, stats in metrics.items():
        print(f"Val {val}: TP={stats['tp']}  FP={stats['fp']}  FN={stats['fn']}  Prec={stats['precision']:.2f}  Rec={stats['recall']:.2f}")


# === MAIN ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('checkpoints/unet_epoch20.pt', map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = RoadDataset("data/thinning/inputs", "data/thinning/targets", transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    with torch.no_grad():
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = torch.sigmoid(model(inputs))

        pred = outputs[0][0].cpu().numpy()
        true = targets[0][0].cpu().numpy()

        # === Clean and Skeletonize ===
        binary_pred, pred_skeleton = clean_and_skeletonize(pred)

        # === Visualize and Print Valences ===
        debug_valence_map(pred_skeleton, label="Prediction")
        debug_valence_map(skeletonize(true > 0.5), label="Target")

        # === Compute Node Metrics ===
        node_metrics = compute_node_metrics(pred_skeleton, true, max_dist=7)
        print("\n📊 Real Sample Metric Results:")
        for val, stats in node_metrics.items():
            print(f"Val {val}: TP={stats['tp']}  FP={stats['fp']}  FN={stats['fn']}  Prec={stats['precision']:.2f}  Rec={stats['recall']:.2f}")

    # === Run Toy Junction Test ===
    toy_junction_test()


if __name__ == "__main__":
    main()
