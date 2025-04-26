import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.unet import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt

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
            axes[2].imshow(outputs[i][0].cpu().numpy() > 0.5, cmap='gray')
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
    num_epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1] tensor
    ])

    dataset = RoadDataset(image_dir, target_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
        save_predictions(model, dataloader, device, epoch)
    
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    main()
