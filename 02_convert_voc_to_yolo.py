import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
xml_dir = "datasets/helmet_dataset/Annotations"
img_dir = "datasets/helmet_dataset/JPEGImages"
output_dir = "datasets/helmet_yolo_clean"

# Create output folders
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

# -----------------------------
# Class mapping
# -----------------------------
class_mapping = {
    "hat": 0,
    "person": 1
}

# -----------------------------
# VOC -> YOLO bbox conversion
# -----------------------------
def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return (
        x_center * dw,
        y_center * dh,
        width * dw,
        height * dh
    )

# -----------------------------
# Collect valid XML files
# -----------------------------
xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
valid_files = []

for xml_file in xml_files:
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    filename = root.find("filename").text

    img_path = os.path.join(img_dir, filename)
    if os.path.exists(img_path):
        valid_files.append(xml_file)

print(f"Total valid annotation-image pairs: {len(valid_files)}")

# -----------------------------
# Split dataset
# -----------------------------
train_files, temp_files = train_test_split(valid_files, test_size=0.2, random_state=42)
valid_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

splits = {
    "train": train_files,
    "valid": valid_files,
    "test": test_files
}

print(f"Train: {len(train_files)}")
print(f"Valid: {len(valid_files)}")
print(f"Test: {len(test_files)}")

# -----------------------------
# Convert files
# -----------------------------
for split_name, files in splits.items():
    for xml_file in files:
        xml_path = os.path.join(xml_dir, xml_file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        img_path = os.path.join(img_dir, filename)

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        yolo_lines = []

        for obj in root.findall("object"):
            cls = obj.find("name").text.strip()

            if cls not in class_mapping:
                continue

            cls_id = class_mapping[cls]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x, y, bw, bh = convert_bbox((w, h), (xmin, ymin, xmax, ymax))
            yolo_lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(output_dir, split_name, "labels", label_filename)

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        shutil.copy(img_path, os.path.join(output_dir, split_name, "images", filename))

print("✅ Dataset converted successfully!")

# -----------------------------
# Create YAML
# -----------------------------
yaml_content = """
path: datasets/helmet_yolo_clean

train: train/images
val: valid/images
test: test/images

nc: 2
names: ['hat', 'person']
"""

with open(os.path.join(output_dir, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("✅ data.yaml created!")

for split in ["train", "valid", "test"]:
    img_count = len(os.listdir(os.path.join(output_dir, split, "images")))
    label_count = len(os.listdir(os.path.join(output_dir, split, "labels")))
    print(f"{split}: {img_count} images, {label_count} labels")