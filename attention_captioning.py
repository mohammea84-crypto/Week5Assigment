# Step 4: Attention-Based Image Captioning using ViT-GPT2
# Commit message: "Implemented attention-based image captioning"

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

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

# Pick a different sample image (index 1)
sample_img = images[1]
true_label = CIFAR10_CLASSES[labels[1]]
image = Image.fromarray(sample_img.astype(np.uint8)).resize((224, 224))
print(f"True CIFAR-10 Label: {true_label}")

# Load ViT-GPT2 captioning model (uses attention mechanism)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Generate caption with attention-based model
inputs = processor(images=image, return_tensors="pt")
caption_ids = model.generate(**inputs, max_length=30)
caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
print(f"Generated Caption with Attention: {caption}")

# --- Output: Save attention-captioned image ---
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(image)
ax.set_title(f"True: {true_label}\nAttention Caption: {caption}", fontsize=9, wrap=True)
ax.axis("off")
plt.tight_layout()
plt.savefig("attention_captioned_image.png", dpi=150)
plt.close()
print("Saved: attention_captioned_image.png")