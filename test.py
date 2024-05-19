import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(1024, 500)

    def forward(self, x):
        return self.fc(x)

# Create a model instance
model = MyModel()

# Check for available GPUs and set up DataParallel
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = nn.DataParallel(model)  # This uses all available GPUs
    model.to(device)
else:
    device = torch.device("cpu")
    model.to(device)

for i in range(torch.cuda.device_count()):
    print(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
    print(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

# Create dummy dataset and dataloader
dataset = TensorDataset(torch.randn(1000, 1024), torch.randint(0, 500, (1000,)))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")

        # I want to check the inputs device on 4 GPUs, or 1 GPUs
        print(inputs.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        for i in range(torch.cuda.device_count()):
            print(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
