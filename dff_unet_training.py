

# dff_unet_training.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt

# Allow PIL to load truncated images (we still skip corrupt ones)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# 1. CONFIG - change these if needed
# -----------------------------
IMAGE_DIR = r"C:\Users\priya\OneDrive\Desktop\Project1\data\images\train"  # <- update if different
MASK_DIR  = r"C:\Users\priya\OneDrive\Desktop\Project1\data\masks\train"    # <- update if different
MODEL_SAVE_PATH = "dff_unet_model.pth"
IMG_SIZE = 128        # smaller for faster CPU runs; increase if you have GPU/time
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 2. Dataset (handles _segmentation masks, skips missing/corrupt)
# -----------------------------
class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # gather image filenames (jpg/png)
        all_images = sorted([f for f in os.listdir(image_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        self.pairs = []
        for img_name in all_images:
            base = os.path.splitext(img_name)[0]
            # ISIC masks usually named like: <base>_segmentation.png
            mask_name = base + "_segmentation.png"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.pairs.append((img_name, mask_name))
            else:
                # if mask missing, skip and print once per missing file
                print(f"⚠️ Skipping {img_name} — mask not found as {mask_name}")

        print(f"✅ {len(self.pairs)} valid image-mask pairs found (out of {len(all_images)} images).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # defensive: wrap in try/except and skip corrupted files gracefully
        img_name, mask_name = self.pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            image = Image.open(img_path).convert("RGB")
            mask  = Image.open(mask_path).convert("L")
        except Exception as e:
            # print a helpful message and choose next item (wrap-around)
            print(f"⚠️ Skipping corrupted file {img_name}: {e}")
            next_idx = (idx + 1) % len(self.pairs)
            return self.__getitem__(next_idx)

        # resize and transform
        if self.transform:
            image = self.transform(image)
            mask  = self.transform(mask)
        else:
            # default transforms: resize + to tensor (0..1)
            t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                    transforms.ToTensor()])
            image = t(image)
            mask  = t(mask)

        # ensure mask is single-channel (shape [1, H, W])
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask[0:1, :, :]

        return image, mask


# -----------------------------
# 3. DFF-UNet model (lightweight, suitable for demo)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class DFF_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Simple "deep feature fusion" by concatenating encoder features at decoder steps
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)         # 64
        e2 = self.enc2(self.pool1(e1))  # 128
        e3 = self.enc3(self.pool2(e2))  # 256
        e4 = self.enc4(self.pool3(e3))  # 512

        b = self.bottleneck(self.pool4(e4))  # 1024

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return self.out_act(out)


# -----------------------------
# 4. Training routine
# -----------------------------
def train_model():
    # setup transforms and dataset
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    dataset = ISICDataset(IMAGE_DIR, MASK_DIR, transform)
    if len(dataset) == 0:
        raise RuntimeError("No valid image-mask pairs found. Check IMAGE_DIR/MASK_DIR and mask naming (_segmentation.png).")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    model = DFF_UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    print(f"Using device: {DEVICE}, training on {len(dataset)} pairs, batch size {BATCH_SIZE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", ncols=100)
        for imgs, masks in loop:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"✅ Epoch {epoch}/{NUM_EPOCHS} finished. Avg Loss: {epoch_loss:.4f}")

    # save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🎉 Training complete — model saved to {MODEL_SAVE_PATH}")

    # plot loss curve
    try:
        plt.figure()
        plt.plot(range(1, NUM_EPOCHS + 1), losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    train_model()
