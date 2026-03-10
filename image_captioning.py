# Step 3: Image Captioning with RNNs using BLIP
# Commit message: "Implemented RNN-based image captioning"

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

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

# Pick a sample image (index 0)
sample_img = images[0]
true_label = CIFAR10_CLASSES[labels[0]]
image = Image.fromarray(sample_img.astype(np.uint8)).resize((224, 224))
print(f"True CIFAR-10 Label: {true_label}")

# Load pretrained BLIP image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Generate caption
inputs = processor(images=image, return_tensors="pt")
caption_ids = model.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)
print(f"Generated Caption: {caption}")

# --- Output: Save captioned image ---
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(image)
ax.set_title(f"True: {true_label}\nCaption: {caption}", fontsize=9, wrap=True)
ax.axis("off")
plt.tight_layout()
plt.savefig("captioned_image.png", dpi=150)
plt.close()
print("Saved: captioned_image.png")