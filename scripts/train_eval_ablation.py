# This is a master training and evaluation runner
# It trains the model (with config), evaluates it visually + quantitatively, and logs everything
# You can then use this same structure to run ablation study sweeps

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import UNet
from models.node_metrics import compute_node_metrics
from skimage.morphology import skeletonize, closing, footprint_rectangle
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
from PIL import Image
from scripts.dataset import RoadDataset  # Assumes you modularized this earlier
from tqdm import tqdm
import sys
sys.path.append("..")  # Adds project root to Python path
from models.unet import UNet
from scripts.dataset import RoadDataset
torch.manual_seed(42)

# Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CONFIG ===
def get_config(name):
    configs = {
        "baseline": {"lr": 1e-3, "loss": nn.BCEWithLogitsLoss(), "epochs": 10},
        "alt_loss": {"lr": 1e-3, "loss": nn.MSELoss(), "epochs": 10},
        "low_lr": {"lr": 1e-4, "loss": nn.BCEWithLogitsLoss(), "epochs": 10},
    }
    return configs[name]


# === UTILS ===
def save_sample_visual(inputs, targets, outputs, out_dir, index):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(inputs[index][0].cpu(), cmap='gray'); axes[0].set_title("Input")
    axes[1].imshow(targets[index][0].cpu(), cmap='gray'); axes[1].set_title("Target")
    pred = torch.sigmoid(outputs[index][0]).cpu().numpy()
    axes[2].imshow(pred > 0.5, cmap='gray'); axes[2].set_title("Prediction")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"qualitative_{index+1}.png"))
    plt.close()


def clean_and_skeletonize(pred):
    pred = pred / pred.max()
    binary_pred = pred > 0.01
    footprint = footprint_rectangle((3, 3))
    binary_pred = closing(binary_pred, footprint)
    return skeletonize(binary_pred)


def save_node_metrics(pred_skeleton, true_target, out_path):
    metrics = compute_node_metrics(pred_skeleton, true_target)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Valence", "Precision", "Recall", "True Positives", "False Positives", "False Negatives"])
        for v, stats in metrics.items():
            writer.writerow([v, f"{stats['precision']:.4f}", f"{stats['recall']:.4f}", stats['tp'], stats['fp'], stats['fn']])


# === EVALUATION (uses fixed sample) ===
def evaluate(model, dataloader, criterion, config_name, results_dir, fixed_batch):
    model.eval()
    os.makedirs(results_dir, exist_ok=True)
    with torch.no_grad():
        inputs, targets = fixed_batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        for i in range(3):
            # Save visual output
            save_sample_visual(inputs, targets, outputs, out_dir=results_dir, index=i)

            # Save per-sample metrics
            sample_dir = os.path.join(results_dir, f"sample_{i+1}")
            os.makedirs(sample_dir, exist_ok=True)

            loss = criterion(outputs[i:i+1], targets[i:i+1]).item()
            with open(os.path.join(sample_dir, "loss.txt"), 'w') as f:
                f.write(f"Loss: {loss:.6f}\n")

            mse = nn.MSELoss()(torch.sigmoid(outputs[i:i+1]), targets[i:i+1]).item()
            with open(os.path.join(sample_dir, "mse.txt"), 'w') as f:
                f.write(f"MSE: {mse:.6f}\n")

            pred_np = torch.sigmoid(outputs[i][0]).cpu().numpy()
            true_np = targets[i][0].cpu().numpy()
            pred_skeleton = clean_and_skeletonize(pred_np)
            save_node_metrics(pred_skeleton, true_np, os.path.join(sample_dir, "node_metrics.csv"))


# === TRAINING ===
def train_and_evaluate(config_name, fixed_batch):
    config = get_config(config_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{config_name}_{timestamp}"

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = RoadDataset("data/thinning/inputs", "data/thinning/targets", transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = config["loss"]

    for epoch in range(config["epochs"]):
        model.train()
        print(f"Training {config_name} - Epoch {epoch+1}/{config['epochs']}")
        running_loss = 0.0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Average Epoch {epoch+1} Loss: {avg_loss:.6f}")

    # Evaluate after training using fixed batch
    evaluate(model, dataloader, criterion, config_name, results_dir, fixed_batch)


# === RUN CONFIGS ===
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    common_dataset = RoadDataset("data/thinning/inputs", "data/thinning/targets", transform)
    common_loader = DataLoader(common_dataset, batch_size=8, shuffle=False)
    fixed_batch = next(iter(common_loader))  # use same batch for all evaluations

    for config in ["baseline", "alt_loss", "low_lr"]:
        train_and_evaluate(config, fixed_batch)
