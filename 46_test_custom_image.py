import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# =====================================================================
# 1. SETTINGS
# =====================================================================
IMAGE_PATH = r"C:\Users\zaabola\Desktop\PPE_Project\datasets\5.png"
MODEL_WEIGHTS = r"C:\Users\zaabola\Desktop\PPE_Project\models\unet_land_segmentation_last.pth"
OUTPUT_DIR = r"C:\Users\zaabola\Desktop\PPE_Project\outputs\unet_test_results"

# UNet requires width/height to be multiples of 32. 
# We use 512x512 to preserve more detail than 256x256
IMAGE_SIZE = (512, 512) 
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
# 3. INFERENCE PIPELINE
# =====================================================================
def test_custom_image():
    if not os.path.exists(IMAGE_PATH):
        print(f"[!] Error: Cannot find image at {IMAGE_PATH}")
        return
        
    model = load_model()
    
    print(f"Processing Custom Image: {os.path.basename(IMAGE_PATH)}")
    
    # 1. Read and Prepare Image
    orig_img = cv2.imread(IMAGE_PATH)
    if orig_img is None:
        print(f"[!] Error: Failed to load image {IMAGE_PATH}")
        return
        
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Resize to multiple of 32
    resized_img = cv2.resize(orig_img, IMAGE_SIZE)
    
    # Normalization (must match training exactly)
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
    overlay = cv2.addWeighted(resized_img, 0.6, pred_rgb, 0.4, 0)
    
    # 5. Save and Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(resized_img)
    axes[0].set_title("Original Gabes Image")
    axes[0].axis('off')
    
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Transparent Overlay")
    axes[2].axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, "custom_gabes_result.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"\nInference complete! Saved result to: {save_path}")

if __name__ == "__main__":
    test_custom_image()
