import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import glob
from tqdm import tqdm
import segmentation_models_pytorch as smp
import random

# =====================================================================
# 1. HYPERPARAMETERS AND SETTINGS
# =====================================================================
DATA_DIR = r"c:\Users\zaabola\Desktop\PPE_Project\datasets\land_segmentation"
MODEL_SAVE_PATH = r"c:\Users\zaabola\Desktop\PPE_Project\models\unet_land_segmentation_best.pth"
FINAL_MODEL_SAVE_PATH = r"c:\Users\zaabola\Desktop\PPE_Project\models\unet_land_segmentation_last.pth"

BATCH_SIZE = 8  # Adjust based on GPU memory
EPOCHS = 30     # Increased for better convergence
LEARNING_RATE = 3e-4 # Good starting LR for pre-trained backbones
IMAGE_SIZE = (256, 256)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

RGB_TO_CLASS = {
    (0, 255, 255): 0,   # urban_land
    (255, 255, 0): 1,   # agriculture_land
    (255, 0, 255): 2,   # rangeland
    (0, 255, 0): 3,     # forest_land
    (0, 0, 255): 4,     # water
    (255, 255, 255): 5, # barren_land
    (0, 0, 0): 6        # unknown
}
NUM_CLASSES = len(RGB_TO_CLASS)

# =====================================================================
# 2. DATASET DEFINITION WITH AUGMENTATION
# =====================================================================
class LandSegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 256)):
        self.split = split
        self.img_size = img_size
        train_dir = os.path.join(root_dir, "train")
        all_images = sorted(glob.glob(os.path.join(train_dir, "*_sat.jpg")))
        
        split_idx = int(len(all_images) * 0.8)
        if split == "train":
            self.images = all_images[:split_idx]
        else:
            self.images = all_images[split_idx:]

    def __len__(self):
        return len(self.images)
        
    def _rgb_to_mask(self, mask_img):
        mask = np.zeros((mask_img.shape[0], mask_img.shape[1]), dtype=np.int64)
        for rgb, idx in RGB_TO_CLASS.items():
            match = np.all(mask_img == np.array(rgb), axis=-1)
            mask[match] = idx
        return mask

    def transform(self, image, mask):
        # Resize
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to class indices
        mask_class = self._rgb_to_mask(mask)
        
        # To PIL for torchvision
        image = TF.to_pil_image(image)
        mask = TF.to_pil_image(mask_class.astype(np.uint8))
        
        # Random Data Augmentation for training only
        if self.split == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        
        # To Tensor
        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        
        # Normalize
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image_tensor, mask_tensor

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = img_path.replace("_sat.jpg", "_mask.png")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        image_tensor, mask_tensor = self.transform(image, mask)
        
        return image_tensor, mask_tensor

# =====================================================================
# 3. TRAINING LOOP
# =====================================================================
def train_model():
    print(f"\nTraining on Device: {DEVICE}")
    print("Loading datasets...")
    train_dataset = LandSegmentationDataset(DATA_DIR, split="train", img_size=IMAGE_SIZE)
    val_dataset = LandSegmentationDataset(DATA_DIR, split="valid", img_size=IMAGE_SIZE)
    
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Use SMP Pretrained UNet
    print("Loading Pretrained ResNet34 UNet...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_loss = float('inf')
    
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}", flush=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved best model to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})", flush=True)
            
    print(f"\nTraining Complete! Saving final model...")
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
    print(f"Final model saved to {FINAL_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
