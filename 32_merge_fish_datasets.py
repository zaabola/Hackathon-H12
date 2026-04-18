"""
Merge both fish datasets into one clean dataset.
- Converts polygon annotations → bounding boxes
- Combines base + fine-tune datasets
- Creates proper train/valid/test splits
"""

import os
import shutil
import random

random.seed(42)

SRC_BASE = "datasets/fish_dataset"
SRC_FINE = "datasets/fish_dataset_fine_tune"
DST = "datasets/fish_merged"


def polygon_to_bbox(parts):
    """Convert polygon annotation (class x1 y1 x2 y2 ...) to bbox (class cx cy w h)."""
    cls_id = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def is_bbox_format(parts):
    """Check if annotation line is in bbox format (5 values: class cx cy w h)."""
    return len(parts) == 5


def convert_label(label_path):
    """Read a label file and return all lines converted to single-class bbox format."""
    with open(label_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    converted = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue  # skip malformed lines
        
        if is_bbox_format(parts):
            # Already bbox format — just remap class to 0
            _, cx, cy, w, h = parts
            converted.append(f"0 {cx} {cy} {w} {h}")
        else:
            # Polygon format — convert to bbox
            converted.append(polygon_to_bbox(parts))
    
    return converted


def copy_dataset(src_dir, dst_dir, split, prefix, stats):
    """Copy images and converted labels from a source split to the merged dataset."""
    img_src = os.path.join(src_dir, split, "images")
    lbl_src = os.path.join(src_dir, split, "labels")
    
    if not os.path.exists(img_src):
        print(f"  ⚠️  {img_src} not found, skipping")
        return
    
    img_dst = os.path.join(dst_dir, split, "images")
    lbl_dst = os.path.join(dst_dir, split, "labels")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)
    
    images = [f for f in os.listdir(img_src) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    
    copied = 0
    skipped = 0
    
    for img_file in images:
        base_name = os.path.splitext(img_file)[0]
        lbl_file = base_name + ".txt"
        lbl_path = os.path.join(lbl_src, lbl_file)
        
        # Skip images without labels
        if not os.path.exists(lbl_path):
            skipped += 1
            continue
        
        # Convert label
        converted_lines = convert_label(lbl_path)
        if not converted_lines:
            skipped += 1
            continue
        
        # Use prefix to avoid name collisions between datasets
        new_name = f"{prefix}_{img_file}"
        new_base = f"{prefix}_{base_name}"
        
        # Copy image
        shutil.copy2(os.path.join(img_src, img_file), os.path.join(img_dst, new_name))
        
        # Write converted label
        with open(os.path.join(lbl_dst, new_base + ".txt"), 'w') as f:
            f.write("\n".join(converted_lines) + "\n")
        
        copied += 1
    
    stats[split] = stats.get(split, 0) + copied
    print(f"  [{split}] Copied {copied} images, Skipped {skipped}")


def main():
    # Clean destination
    if os.path.exists(DST):
        shutil.rmtree(DST)
    
    print("=" * 60)
    print("  Merging Fish Datasets")
    print("=" * 60)
    
    stats = {}
    
    # Copy base dataset
    print(f"\n📁 Source: {SRC_BASE}")
    for split in ["train", "valid", "test"]:
        copy_dataset(SRC_BASE, DST, split, "base", stats)
    
    # Copy fine-tune dataset
    print(f"\n📁 Source: {SRC_FINE}")
    for split in ["train", "valid", "test"]:
        copy_dataset(SRC_FINE, DST, split, "fine", stats)
    
    # Create data.yaml
    yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['fish']
"""
    yaml_path = os.path.join(DST, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'=' * 60}")
    print(f"  ✅ Merged Dataset Created: {DST}")
    print(f"{'=' * 60}")
    for split, count in stats.items():
        print(f"  {split}: {count} images")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Classes: 1 (fish)")
    
    # Verify a sample label
    sample_dir = os.path.join(DST, "train", "labels")
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.txt')][:3]
    print(f"\n  Sample labels:")
    for f in sample_files:
        with open(os.path.join(sample_dir, f)) as fh:
            content = fh.read().strip()
        print(f"    {f}: {content[:80]}...")


if __name__ == "__main__":
    main()
