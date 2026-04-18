import os
import xml.etree.ElementTree as ET
import shutil
import random

# ==========================================
# PATHS
# ==========================================
DATASET_PATH = "datasets/mask_fine_tunning"
OUTPUT_PATH = "datasets/mask_yolo_finetune"

# IMPORTANT:
# Kaggle dataset class names
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================
os.makedirs(f"{OUTPUT_PATH}/train/images", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/train/labels", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/valid/images", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/valid/labels", exist_ok=True)

# ==========================================
# CONVERT VOC BOX -> YOLO BOX
# ==========================================
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh

    return x, y, w, h

# ==========================================
# GET ALL XML FILES
# ==========================================
xml_files = [f for f in os.listdir(f"{DATASET_PATH}/annotations") if f.endswith(".xml")]
random.shuffle(xml_files)

# 80% train / 20% valid
split_idx = int(0.8 * len(xml_files))

print(f"📦 Found {len(xml_files)} annotation files")
print(f"➡ Train: {split_idx}")
print(f"➡ Valid: {len(xml_files) - split_idx}")

# ==========================================
# PROCESS EACH XML
# ==========================================
for i, xml_file in enumerate(xml_files):
    xml_path = os.path.join(DATASET_PATH, "annotations", xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(DATASET_PATH, "images", filename)

    if not os.path.exists(img_path):
        print(f"⚠ Image not found: {img_path}")
        continue

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    txt_filename = xml_file.replace(".xml", ".txt")

    # split
    if i < split_idx:
        out_img_path = os.path.join(OUTPUT_PATH, "train", "images", filename)
        out_label_path = os.path.join(OUTPUT_PATH, "train", "labels", txt_filename)
    else:
        out_img_path = os.path.join(OUTPUT_PATH, "valid", "images", filename)
        out_label_path = os.path.join(OUTPUT_PATH, "valid", "labels", txt_filename)

    # copy image
    shutil.copy(img_path, out_img_path)

    # write YOLO label
    with open(out_label_path, "w") as f:
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()

            if cls_name not in CLASSES:
                print(f"⚠ Unknown class '{cls_name}' in {xml_file}")
                continue

            cls_id = CLASSES.index(cls_name)

            xmlbox = obj.find("bndbox")
            xmin = float(xmlbox.find("xmin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymin = float(xmlbox.find("ymin").text)
            ymax = float(xmlbox.find("ymax").text)

            yolo_box = convert((width, height), (xmin, xmax, ymin, ymax))
            f.write(f"{cls_id} {' '.join(map(str, yolo_box))}\n")

# ==========================================
# CREATE data.yaml
# ==========================================
yaml_content = """path: datasets/mask_yolo_finetune

train: train/images
val: valid/images

nc: 3
names: ['with_mask', 'without_mask', 'incorrect_mask']
"""

with open(os.path.join(OUTPUT_PATH, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("\n✅ Conversion complete!")
print(f"📁 YOLO dataset saved to: {OUTPUT_PATH}")
print("📄 data.yaml created successfully!")