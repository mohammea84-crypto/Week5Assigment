# Step 6: Neural Style Transfer using VGG19
# Commit message: "Implemented neural style transfer"

import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Image loader helper
def load_image(path, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

# Gram matrix for style loss
def gram_matrix(feature_map):
    _, C, H, W = feature_map.size()
    features = feature_map.view(C, H * W)
    return torch.mm(features, features.t()) / (C * H * W)

# Content and style loss functions
def content_loss(target, content):
    return torch.mean((target - content) ** 2)

def style_loss(target, style):
    return torch.mean((gram_matrix(target) - gram_matrix(style)) ** 2)

# Load content and style images (using same image for demo simplicity)
image_path = "sample.jpg"
content_image = load_image(image_path)
style_image = load_image(image_path)   # Replace with a style image if available

# Use VGG19 feature extractor
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# Target image starts as a copy of content image
target = content_image.clone().requires_grad_(True)
optimizer = optim.Adam([target], lr=0.01)

# Optimization loop (reduced iterations for fast assignment run)
print("Running style transfer (200 iterations)...")
for i in range(200):
    target_feat = vgg(target)
    content_feat = vgg(content_image).detach()
    style_feat = vgg(style_image).detach()

    c_loss = content_loss(target_feat, content_feat)
    s_loss = style_loss(target_feat, style_feat)
    loss = c_loss + 1e4 * s_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 50 == 0:
        print(f"  Iteration {i+1}/200 | Loss: {loss.item():.4f}")

# --- Output: Save stylized image vs original side-by-side ---
original = Image.open(image_path).convert("RGB").resize((256, 256))
stylized = target.squeeze().detach().clamp(0, 1).permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(original)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(stylized)
axes[1].set_title("Style Transferred Image")
axes[1].axis("off")
plt.suptitle("Neural Style Transfer Result", fontsize=13)
plt.tight_layout()
plt.savefig("style_transfer_result.png", dpi=150)
plt.close()
print("Saved: style_transfer_result.png")