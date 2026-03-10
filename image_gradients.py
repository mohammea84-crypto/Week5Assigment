# Step 5: Image Gradients - Saliency Maps & Top-5 Predictions
# Commit message: "Generated saliency maps for image visualization"

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

# Load one batch from CIFAR-10 dataset
def load_cifar10_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch[b'labels']
    return images, labels

# Load data_batch_1 from local dataset folder
data_path = os.path.join("data", "cifar-10-batches-py", "data_batch_1")
images, labels = load_cifar10_batch(data_path)

# Pick a sample image (index 2)
sample_img = images[2]
true_label = CIFAR10_CLASSES[labels[2]]
raw_image = Image.fromarray(sample_img.astype(np.uint8)).resize((224, 224))
print(f"True CIFAR-10 Label: {true_label}")

# Preprocess for ResNet18
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensor = transform(raw_image).unsqueeze(0)
image_tensor.requires_grad_(True)

# Load pretrained ResNet18 and run forward pass
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
output = model(image_tensor)
class_idx = torch.argmax(output).item()
output[0, class_idx].backward()

# --- Output 1: Saliency Map (original + heatmap) ---
saliency_map = image_tensor.grad.abs().squeeze().max(dim=0)[0].detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(raw_image)
axes[0].set_title(f"CIFAR-10 Image\nTrue Label: {true_label}")
axes[0].axis("off")
axes[1].imshow(saliency_map, cmap="hot")
axes[1].set_title("Saliency Map")
axes[1].axis("off")
plt.suptitle("Image Gradient Visualization", fontsize=13)
plt.tight_layout()
plt.savefig("saliency_map.png", dpi=150)
plt.close()
print("Saved: saliency_map.png")

# --- Output 2: Confusion Matrix across 5 CIFAR-10 samples ---
# Run 5 images through ResNet18 and compare true vs predicted CIFAR-10 label index
true_indices = []
pred_indices = []

for i in range(25):  # use 25 samples for a richer confusion matrix
    img = Image.fromarray(images[i].astype(np.uint8)).resize((224, 224))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    pred_imagenet = torch.argmax(out).item()
    # Map ImageNet prediction to nearest CIFAR-10 class by index mod 10
    pred_cifar = pred_imagenet % 10
    true_indices.append(labels[i])
    pred_indices.append(pred_cifar)

# Build 10x10 confusion matrix
conf_matrix = np.zeros((10, 10), dtype=int)
for t, p in zip(true_indices, pred_indices):
    conf_matrix[t][p] += 1

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf_matrix, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix (ResNet18 on CIFAR-10 Samples)")
for i in range(10):
    for j in range(10):
        ax.text(j, i, conf_matrix[i, j], ha="center", va="center", fontsize=8)
plt.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# Print top-5 predictions for the sample image
probs = torch.softmax(output.detach(), dim=1).squeeze()
top5_probs, top5_ids = probs.topk(5)
print(f"\nTop-5 ImageNet Predictions for '{true_label}':")
print(f"{'Rank':<6}{'Class Index':<15}{'Confidence':>12}")
print("-" * 33)
for rank, (idx, prob) in enumerate(zip(top5_ids, top5_probs), 1):
    print(f"{rank:<6}{idx.item():<15}{prob.item()*100:>10.2f}%")