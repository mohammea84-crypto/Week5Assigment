# Step 6: Neural Style Transfer using VGG19
# Commit message: "Implemented neural style transfer"

import os
import pickle
import torch
import torch.optim as optim
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Load one batch from CIFAR-10 dataset
def load_cifar10_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch[b'labels']
    return images, labels

# Convert CIFAR-10 numpy image to tensor
def to_tensor(np_img, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(Image.fromarray(np_img.astype(np.uint8))).unsqueeze(0)

# Load two different CIFAR-10 images: content (index 3) and style (index 4)
data_path = os.path.join("data", "cifar-10-batches-py", "data_batch_1")
images, labels = load_cifar10_batch(data_path)

content_image = to_tensor(images[3])
style_image   = to_tensor(images[4])

# Gram matrix for style loss
def gram_matrix(feature_map):
    _, C, H, W = feature_map.size()
    features = feature_map.view(C, H * W)
    return torch.mm(features, features.t()) / (C * H * W)

def content_loss(target, content):
    return torch.mean((target - content) ** 2)

def style_loss(target, style):
    return torch.mean((gram_matrix(target) - gram_matrix(style)) ** 2)

# Load VGG19 feature extractor
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# Target image starts as content image
target = content_image.clone().requires_grad_(True)
optimizer = optim.Adam([target], lr=0.01)

# Optimization loop
print("Running style transfer (200 iterations)...")
for i in range(200):
    target_feat  = vgg(target)
    content_feat = vgg(content_image).detach()
    style_feat   = vgg(style_image).detach()

    loss = content_loss(target_feat, content_feat) + 1e4 * style_loss(target_feat, style_feat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 50 == 0:
        print(f"  Iteration {i+1}/200 | Loss: {loss.item():.4f}")

# --- Output: Save content / style / stylized side-by-side ---
content_show = content_image.squeeze().permute(1, 2, 0).detach().numpy()
style_show   = style_image.squeeze().permute(1, 2, 0).detach().numpy()
stylized     = target.squeeze().detach().clamp(0, 1).permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(content_show); axes[0].set_title("Content Image"); axes[0].axis("off")
axes[1].imshow(style_show);   axes[1].set_title("Style Image");   axes[1].axis("off")
axes[2].imshow(stylized);     axes[2].set_title("Stylized Output"); axes[2].axis("off")
plt.suptitle("Neural Style Transfer (CIFAR-10)", fontsize=13)
plt.tight_layout()
plt.savefig("style_transfer_result.png", dpi=150)
plt.close()
print("Saved: style_transfer_result.png")