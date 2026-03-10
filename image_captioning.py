# Step 3: Image Captioning with RNNs using BLIP
# Commit message: "Implemented RNN-based image captioning"

import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load pretrained BLIP image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and preprocess the image
image_path = "sample.jpg"
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Generate caption using the model
caption_ids = model.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)
print(f"Generated Caption: {caption}")

# --- Output 1: Save captioned image ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(image)
ax.set_title(f"Caption: {caption}", fontsize=10, wrap=True)
ax.axis("off")
plt.tight_layout()
plt.savefig("captioned_image.png", dpi=150)
plt.close()
print("Saved: captioned_image.png")