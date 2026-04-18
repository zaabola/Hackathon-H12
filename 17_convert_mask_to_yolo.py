import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import cv2

BASE_DIR = "datasets/mask_dataset/Train"
IMG_DIR = os.path.join(BASE_DIR, "Pictures")
XML_DIR = os.path.join(BASE_DIR, "Annotations")

OUTPUT_DIR = "datasets/mask_yolo_clean"

class_names = ["with_mask", "without_mask", "mask_weared_incorrect"]

# Create folders
for split in ["train", "valid", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

# Convert XML → YOLO
def convert_xml(xml_path, w, h):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []

    for obj in root.findall("object"):
        cls = obj.find("name").text

        if cls not in class_names:
            continue

        cls_id = class_names.index(cls)

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        lines.append(f"{cls_id} {x_center} {y_center} {width} {height}")

    return lines

# Get all images
images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]

# Split dataset
train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
valid_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

splits = {
    "train": train_imgs,
    "valid": valid_imgs,
    "test": test_imgs
}

# Process
for split, img_list in splits.items():
    for img in img_list:
        img_path = os.path.join(IMG_DIR, img)
        xml_path = os.path.join(XML_DIR, img.replace(".jpg", ".xml"))

        if not os.path.exists(xml_path):
            continue

        image = cv2.imread(img_path)
        h, w, _ = image.shape

        lines = convert_xml(xml_path, w, h)

        if len(lines) == 0:
            continue

        shutil.copy(img_path, f"{OUTPUT_DIR}/{split}/images/{img}")

        with open(f"{OUTPUT_DIR}/{split}/labels/{img.replace('.jpg', '.txt')}", "w") as f:
            f.write("\n".join(lines))

# YAML
yaml_content = f"""
path: datasets/mask_yolo_clean

train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""

with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

print("✅ Dataset fixed and converted properly!")