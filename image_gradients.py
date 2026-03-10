# Step 5: Image Gradients - Saliency Maps & Class Comparison Table
# Commit message: "Generated saliency maps for image visualization"

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ImageNet top-5 class labels (short list for display)
IMAGENET_CLASSES = {
    0: "tench", 1: "goldfish", 2: "great white shark", 3: "tiger shark",
    4: "hammerhead", 5: "electric ray", 283: "Persian cat", 285: "Egyptian cat",
    340: "zebra", 386: "African elephant", 281: "tabby cat",
}

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = "sample.jpg"
raw_image = Image.open(image_path).convert("RGB")
image_tensor = transform(raw_image).unsqueeze(0)
image_tensor.requires_grad_(True)

# Forward pass and compute gradients
output = model(image_tensor)
class_idx = torch.argmax(output).item()
output[0, class_idx].backward()

# --- Output 1: Saliency Map ---
saliency = image_tensor.grad.abs().squeeze()
saliency_map = saliency.max(dim=0)[0].detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(raw_image.resize((224, 224)))
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(saliency_map, cmap="hot")
axes[1].set_title("Saliency Map")
axes[1].axis("off")
plt.suptitle("Image Gradient Visualization", fontsize=13)
plt.tight_layout()
plt.savefig("saliency_map.png", dpi=150)
plt.close()
print("Saved: saliency_map.png")

# --- Output 2: Top-5 Predictions Table (bar chart) ---
probs = torch.softmax(output.detach(), dim=1).squeeze()
top5_probs, top5_ids = probs.topk(5)
top5_labels = [IMAGENET_CLASSES.get(i.item(), f"class_{i.item()}") for i in top5_ids]
top5_scores = [round(p.item() * 100, 2) for p in top5_probs]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(top5_labels[::-1], top5_scores[::-1], color="steelblue")
ax.set_xlabel("Confidence (%)")
ax.set_title("Top-5 Predicted Classes (ResNet18)")
for bar, score in zip(bars, top5_scores[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{score}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("top5_predictions.png", dpi=150)
plt.close()
print("Saved: top5_predictions.png")

# Print top-5 as table
print("\nTop-5 Predictions:")
print(f"{'Rank':<6}{'Class':<25}{'Confidence':>12}")
print("-" * 43)
for rank, (label, score) in enumerate(zip(top5_labels, top5_scores), 1):
    print(f"{rank:<6}{label:<25}{score:>11.2f}%")