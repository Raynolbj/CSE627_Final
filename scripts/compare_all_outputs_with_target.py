import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# === Configuration paths ===
checkpoint_paths = {
    "Baseline": "results/baseline_20250501_205525/model.pth",
    "Alt Loss": "results/alt_loss_20250501_214140/model.pth",
    "Low LR": "results/low_lr_20250501_222546/model.pth",
}
# === Load fixed input and target
transform = transforms.Compose([transforms.ToTensor()])
dataset = RoadDataset("data/thinning/inputs", "data/thinning/targets", transform)
input_tensor, target_tensor = dataset[1]  # Select same image
input_tensor = input_tensor.unsqueeze(0)  # [1, 1, H, W]
target_tensor = target_tensor.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Predict and compare
outputs = {}
metrics_summary = {}

for name, ckpt_path in checkpoint_paths.items():
    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        inp = input_tensor.to(device)
        out = model(inp)
        outputs[name] = torch.sigmoid(out[0][0].cpu()).numpy()

    # Metrics
    pred_bin = (outputs[name] > 0.5).astype(np.uint8)
    target_np = target_tensor[0][0].numpy()

    loss = F.binary_cross_entropy(torch.tensor(outputs[name]), torch.tensor(target_np)).item()
    mse = F.mse_loss(torch.tensor(outputs[name]), torch.tensor(target_np)).item()
    valence_stats = compute_node_metrics(pred_bin, (target_np > 0.5).astype(np.uint8))

    metrics_summary[name] = {
        "loss": loss,
        "mse": mse,
        "valence": valence_stats
    }

# === Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, name in enumerate(list(outputs.keys()) + ["Target"]):
    if name == "Target":
        img = target_tensor[0][0].numpy()
    else:
        img = outputs[name] > 0.5
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(name)
    axes[i].axis("off")

plt.subplots_adjust(top=0.85, wspace=0.1)
plt.tight_layout()
plt.savefig("comparison_same_input.png", bbox_inches='tight')
plt.close()

# === Save CSV
with open("summary_same_input.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Config", "Loss", "MSE", "Valence", "Precision", "Recall", "TP", "FP", "FN"])
    for config, data in metrics_summary.items():
        loss, mse, val_stats = data["loss"], data["mse"], data["valence"]
        for val in range(1, 5):
            s = val_stats.get(val, {"precision": 0, "recall": 0, "tp": 0, "fp": 0, "fn": 0})
            writer.writerow([
                config, f"{loss:.6f}", f"{mse:.6f}", val,
                f"{s['precision']:.4f}", f"{s['recall']:.4f}",
                s["tp"], s["fp"], s["fn"]
            ])

print("✅ Saved comparison_same_input.png and summary_same_input.csv")