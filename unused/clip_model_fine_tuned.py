import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel

# Define your dataset
class MyDataset(Dataset):
    def __init__(self, image_paths, captions, processor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        caption = self.captions[idx]
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding=True)
        return inputs

# Load your image paths and captions
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
captions = ["caption for image 1", "caption for image 2", ...]

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Create the dataset and dataloader
dataset = MyDataset(image_paths, captions, processor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-6)
loss_fn = nn.CrossEntropyLoss()

# Fine-tuning loop
model.train()
num_epochs = 3

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in processor.feature_extractor.model_input_names}

        # Forward pass
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        ground_truth = torch.arange(len(inputs["input_ids"]), device=model.device)

        # Compute the loss
        loss = (loss_fn(logits_per_image, ground_truth) + loss_fn(logits_per_text, ground_truth)) / 2

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("path/to/save/fine_tuned_model")
processor.save_pretrained("path/to/save/fine_tuned_model")



# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel

# # Load the pre-trained CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Load and preprocess the image
# image_path = "path/to/your/image.jpg"
# image = Image.open(image_path)
# inputs = processor(images=image, return_tensors="pt")

# # Encode the image
# with torch.no_grad():
#     image_features = model.get_image_features(**inputs)

# # Normalize the features
# image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# print("Encoded image features:", image_features)
