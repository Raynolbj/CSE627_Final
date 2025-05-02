import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.unet import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.node_metrics import compute_node_metrics
import csv
from datetime import datetime
from skimage.morphology import skeletonize, opening, footprint_rectangle
from models.unet import UNet
from scripts.dataset import RoadDataset

# Dataset
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

        image = Image.open(image_path).convert("L")  # grayscale
        target = Image.open(target_path).convert("L")

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

# Main training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def save_predictions(model, dataloader, device, epoch, out_dir="predictions"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = torch.sigmoid(model(inputs))  # convert logits to probability
        
        for i in range(min(5, inputs.size(0))):  # save first 4 images
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes[0].imshow(inputs[i][0].cpu(), cmap='gray')
            axes[0].set_title("Input")
            axes[1].imshow(targets[i][0].cpu(), cmap='gray')
            axes[1].set_title("Target")
            from skimage.morphology import skeletonize

            pred = (outputs[i][0].cpu().numpy()) > 0.3  # Lower threshold
            pred_skeleton = skeletonize(pred)
            axes[2].imshow(pred_skeleton, cmap='gray')
            axes[2].set_title("Prediction")

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"epoch{epoch+1}_sample{i+1}.png"))
            plt.close()

def main():
    # Configs
    image_dir = "data/thinning/inputs"
    target_dir = "data/thinning/targets"
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1] tensor
    ])

    dataset = RoadDataset(image_dir, target_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    pos_weight = torch.tensor([10.0]).to(device)  # Boost positive class 10x
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        loss = train(model, dataloader, criterion, optimizer, device)
        
        # After training each epoch: (optional) Save predictions
        save_predictions(model, dataloader, device, epoch)

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch{epoch+1}.pt"))

        # Evaluate quick MSE (optional logging)
        model.eval()
        with torch.no_grad():
            inputs, targets = next(iter(dataloader))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.sigmoid(model(inputs))
            mse_loss = mse_criterion(outputs, targets)
        print(f"Epoch {epoch+1}/{num_epochs}, BCE Loss: {loss:.4f}, MSE: {mse_loss.item():.4f}")

    
    # After all epochs are done
    # Final Node Precision & Recall Evaluation
    print("\nEvaluating Node Precision & Recall on a sample batch...")

    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = torch.sigmoid(model(inputs))

        # Take the first sample
        pred = outputs[0][0].cpu().numpy()
        true = targets[0][0].cpu().numpy()

        # Normalize prediction
        pred = pred / pred.max()

        # Improved thresholding and denoising
        binary_pred = pred > 0.05  # Lower threshold
        footprint = footprint_rectangle((3,3))
        binary_pred = opening(binary_pred, footprint)  # Morphological opening to clean noise

        # Skeletonize
        pred_skeleton = skeletonize(binary_pred)

        # Optional: visualize
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(inputs[0][0].cpu(), cmap='gray')
        axes[0].set_title('Input')
        axes[1].imshow(true, cmap='gray')
        axes[1].set_title('Target')
        axes[2].imshow(binary_pred, cmap='gray')
        axes[2].set_title('Thresholded & Opened')
        axes[3].imshow(pred_skeleton, cmap='gray')
        axes[3].set_title('Skeletonized Prediction')

        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

        # Compute node metrics
        node_metrics = compute_node_metrics(pred_skeleton, true)

        # Save node metrics to CSV
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(results_dir, f"node_metrics_eval_{timestamp}.csv")

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Valence", "Precision", "Recall", "True Positives", "False Positives", "False Negatives"])
            for valence, stats in node_metrics.items():
                writer.writerow([
                    valence,
                    f"{stats['precision']:.4f}",
                    f"{stats['recall']:.4f}",
                    stats['tp'],
                    stats['fp'],
                    stats['fn']
                ])

        print(f"\nNode metrics saved to {csv_path}")
    
if __name__ == "__main__":
    main()
