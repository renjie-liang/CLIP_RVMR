import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess the image
image_path = "CLIP4Clip.png"
image = cv2.imread(image_path)
image = torch.tensor(image)
inputs = processor(images=[image, image, image], return_tensors="pt")

# Encode the image
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# Normalize the features
image_features = image_features / image_features.norm(dim=-1, keepdim=True)



texts = ["A photo of a cat", "A photo of a dog", "A photo of a beautiful sunset"]

# Preprocess the text
inputs = processor(text=texts, return_tensors="pt", padding=True)
# Encode the text
with torch.no_grad():
    text_features = model.get_text_features(**inputs)

# Normalize the features
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
