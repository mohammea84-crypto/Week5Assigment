# Step 4: Attention-Based Image Captioning using ViT-GPT2
# Commit message: "Implemented attention-based image captioning"

import matplotlib.pyplot as plt
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load ViT-GPT2 captioning model (uses attention mechanism)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load and preprocess the image
image_path = "sample.jpg"
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Generate caption with attention-based model
caption_ids = model.generate(**inputs, max_length=30)
caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
print(f"Generated Caption with Attention: {caption}")

# --- Output: Save attention-captioned image ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(image)
ax.set_title(f"Attention Caption: {caption}", fontsize=10, wrap=True)
ax.axis("off")
plt.tight_layout()
plt.savefig("attention_captioned_image.png", dpi=150)
plt.close()
print("Saved: attention_captioned_image.png")