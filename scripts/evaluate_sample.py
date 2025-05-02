import sys
import os
import time
import torch
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from skimage.morphology import skeletonize, closing, footprint_rectangle

# === Add parent directory to path for module imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.unet import UNet
from models.node_metrics import compute_node_metrics

# === Dataset class ===
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

# === Gentle cleaning version ===
def clean_and_skeletonize(pred):
    pred = pred / pred.max()
    binary_pred = pred > 0.01  # soft threshold
    footprint = footprint_rectangle((3, 3))
    binary_pred = closing(binary_pred, footprint)
    pred_skeleton = skeletonize(binary_pred)
    return binary_pred, pred_skeleton

# === Main ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('checkpoints/unet_epoch20.pt', map_location=device))
    model.eval()

    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = RoadDataset("data/thinning/inputs", "data/thinning/targets", transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    with torch.no_grad():
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = torch.sigmoid(model(inputs))

        # Use first image in batch
        pred = outputs[0][0].cpu().numpy()
        true = targets[0][0].cpu().numpy()
        input_img = inputs[0][0].cpu().numpy()

        print("Running skeletonization...")
        binary_pred, pred_skeleton = clean_and_skeletonize(pred)

        # === Create a comparison image with titles ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"sample_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["Input", "Target", "Prediction"]
        images = [input_img, true, pred_skeleton]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=12, pad=10)
            ax.axis("off")

        plt.tight_layout()
        comparison_path = os.path.join(results_dir, "comparison.png")
        plt.savefig(comparison_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Comparison image saved to: {comparison_path}")

        # === Node Metric Evaluation with Progress ===
        print("Evaluating node metrics...")
        start = time.time()
        node_metrics = compute_node_metrics(pred_skeleton, true, max_dist=2)
        print(f"✅ Node metrics computed in {time.time() - start:.2f} seconds")

        # === Save metrics to CSV ===
        csv_path = os.path.join(results_dir, "node_metrics.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Valence", "Precision", "Recall", "True Positives", "False Positives", "False Negatives"])
            for valence, stats in tqdm(node_metrics.items(), desc="Writing CSV"):
                writer.writerow([
                    valence,
                    f"{stats['precision']:.4f}",
                    f"{stats['recall']:.4f}",
                    stats['tp'],
                    stats['fp'],
                    stats['fn']
                ])

        print(f"✅ Node metrics saved to: {csv_path}")

if __name__ == "__main__":
    main()
