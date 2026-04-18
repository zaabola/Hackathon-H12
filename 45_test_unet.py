import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import glob
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# =====================================================================
# 1. HYPERPARAMETERS AND SETTINGS
# =====================================================================
TEST_DIR = r"c:\Users\zaabola\Desktop\PPE_Project\datasets\land_segmentation\train"
MODEL_WEIGHTS = r"c:\Users\zaabola\Desktop\PPE_Project\models\unet_land_segmentation_last.pth"
OUTPUT_DIR = r"c:\Users\zaabola\Desktop\PPE_Project\outputs\unet_test_results"

IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_TO_RGB = {
    0: (0, 255, 255),   # urban_land
    1: (255, 255, 0),   # agriculture_land
    2: (255, 0, 255),   # rangeland
    3: (0, 255, 0),     # forest_land
    4: (0, 0, 255),     # water
    5: (255, 255, 255), # barren_land
    6: (0, 0, 0)        # unknown
}
NUM_CLASSES = len(CLASS_TO_RGB)

# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def load_model():
    print(f"Loading Model from {MODEL_WEIGHTS}...")
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"[!] Warning: Model weights not found at {MODEL_WEIGHTS}.")
        return None
        
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def index_to_rgb(mask_idx):
    h, w = mask_idx.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in CLASS_TO_RGB.items():
        rgb_mask[mask_idx == idx] = rgb
    return rgb_mask

# =====================================================================
# 3. TESTING PIPELINE
# =====================================================================
def test_unet():
    model = load_model()
    if model is None:
        return
        
    # Pick a few random images to test
    test_images = glob.glob(os.path.join(TEST_DIR, "*_sat.jpg"))
    if not test_images:
        print(f"No images found in {TEST_DIR}!")
        return
        
    np.random.shuffle(test_images)
    samples = test_images[:5]
    
    print(f"Running inference on {len(samples)} random test images...")
    
    for img_path in samples:
        filename = os.path.basename(img_path)
        print(f"Processing: {filename}")
        
        # 1. Read and Prepare Image and Original Mask
        mask_path = img_path.replace("_sat.jpg", "_mask.png")
        
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(orig_img, IMAGE_SIZE)
        
        orig_mask = cv2.imread(mask_path)
        orig_mask = cv2.cvtColor(orig_mask, cv2.COLOR_BGR2RGB)
        resized_mask = cv2.resize(orig_mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # Exact same normalization as training
        image_tensor = TF.to_tensor(resized_img)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        
        # 2. Run Inference
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        # 3. Convert Prediction back to RGB
        pred_rgb = index_to_rgb(pred_mask)
        
        # 4. Create Transparent Overlay
        # alpha=0.6 for original image, beta=0.4 for mask
        overlay = cv2.addWeighted(resized_img, 0.6, pred_rgb, 0.4, 0)
        
        # 5. Save Side-by-Side Result (1x4)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(resized_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(resized_mask)
        axes[1].set_title("Original Mask (Ground Truth)")
        axes[1].axis('off')
        
        axes[2].imshow(pred_rgb)
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')
        
        axes[3].imshow(overlay)
        axes[3].set_title("Predicted Overlay")
        axes[3].axis('off')
        
        save_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    print(f"\nTesting complete! Visual results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_unet()
