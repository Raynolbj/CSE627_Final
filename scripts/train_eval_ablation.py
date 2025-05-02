# train_eval_ablation.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
from dataset import RoadDataset
from tqdm import tqdm
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(name):
    configs = {
        "baseline": {"lr": 1e-3, "loss": nn.BCEWithLogitsLoss(), "epochs": 10},
        "alt_loss": {"lr": 1e-3, "loss": nn.MSELoss(), "epochs": 10},
        "low_lr": {"lr": 1e-4, "loss": nn.BCEWithLogitsLoss(), "epochs": 10},
    }
    return configs[name]

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
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            fp = max(0, fp)
            fn = max(0, fn)
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            writer.writerow([v, f"{precision:.4f}", f"{recall:.4f}", tp, fp, fn])

def evaluate(model, dataloader, criterion, config_name, results_dir, fixed_batch):
    model.eval()
    os.makedirs(results_dir, exist_ok=True)
    with torch.no_grad():
        inputs, targets = fixed_batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        torch.save(outputs.cpu(), os.path.join(results_dir, "raw_outputs.pt"))  # Debugging

        for i in range(3):
            save_sample_visual(inputs, targets, outputs, out_dir=results_dir, index=i)

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

def train_and_evaluate(config_name):
    print(f"\n==== Starting configuration: {config_name} ====")
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
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1} [{config_name}]"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"[Config: {config_name}] Average Epoch {epoch+1} Loss: {avg_loss:.6f}")

    os.makedirs(results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
    
    # Use fresh fixed batch here
    fixed_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    fixed_batch = next(iter(fixed_loader))

    evaluate(model, dataloader, criterion, config_name, results_dir, fixed_batch)

if __name__ == "__main__":
    for config in ["baseline", "alt_loss", "low_lr"]:
        train_and_evaluate(config)
