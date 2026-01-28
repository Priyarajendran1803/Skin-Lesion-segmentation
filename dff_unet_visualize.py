# dff_unet_visualize.py
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from dff_unet_training import DFF_UNet, IMG_SIZE, DEVICE  # reuse model definition

# ---- PATH CONFIG ----
IMAGE_DIR = r"C:\Users\priya\OneDrive\Desktop\Project1\data\images\train"    # same as training
MASK_DIR  = r"C:\Users\priya\OneDrive\Desktop\Project1\data\masks\train"
MODEL_PATH = "dff_unet_model.pth"

# ---- Load trained model ----
model = DFF_UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# ---- Transform (same as training) ----
transform_img = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---- Utility to show comparison ----
def visualize_prediction(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask  = Image.open(mask_path).convert("L")

    img_tensor = transform_img(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = pred.squeeze().cpu().numpy()

    # show 3 columns: Original | Ground Truth | Predicted
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Predicted Mask")

    for a in axs:
        a.axis("off")
    plt.tight_layout()
    plt.show()


# ---- Visualize few test images ----
# pick first 5 valid pairs
count = 0
for img_name in os.listdir(IMAGE_DIR):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        base = os.path.splitext(img_name)[0]
        mask_name = base + "_segmentation.png"
        mask_path = os.path.join(MASK_DIR, mask_name)
        img_path  = os.path.join(IMAGE_DIR, img_name)

        if os.path.exists(mask_path):
            print(f"\n📸 Showing result for: {img_name}")
            visualize_prediction(img_path, mask_path)
            count += 1
            if count >= 5:  # show first 5
                break
