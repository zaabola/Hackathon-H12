import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

# -----------------------------
# Dataset class for YOLO labels
# -----------------------------
class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, bw, bh = map(float, line.strip().split())

                    xmin = (x - bw / 2) * w
                    ymin = (y - bh / 2) * h
                    xmax = (x + bw / 2) * w
                    ymax = (y + bh / 2) * h

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(cls) + 1)  # background = 0

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        image = F.to_tensor(image)
        return image, target

# -----------------------------
# Collate function
# -----------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

# -----------------------------
# Main training
# -----------------------------
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    train_dataset = YOLODataset(
        "datasets/gas_mask_dataset/train/images",
        "datasets/gas_mask_dataset/train/labels"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    num_classes = 4  # background + Oxygen_tube + gasmask + person

    model = ssd300_vgg16(weights="DEFAULT")
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels=[512, 1024, 512, 256, 256, 256],
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    os.makedirs("models/ssd", exist_ok=True)
    torch.save(model.state_dict(), "models/ssd/ssd_gasmask_5epochs.pth")
    print("✅ SSD gas mask model saved to models/ssd/ssd_gasmask_5epochs.pth")

if __name__ == "__main__":
    main()