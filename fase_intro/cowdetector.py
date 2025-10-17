import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
import os
from PIL import Image
import matplotlib.pyplot as plt

# Your dataset path
data_path = r"C:\Users\dylan\.cache\kagglehub\datasets\alessiocorrado99\animals10\versions\2\raw-img"

# Let's see the folder structure
print("Dataset structure:")
for animal in os.listdir(data_path):
    animal_path = os.path.join(data_path, animal)
    if os.path.isdir(animal_path):
        image_count = len([f for f in os.listdir(animal_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {animal}: {image_count} images")

# Let's look at a sample from each animal type
print("\nLet's look at sample images:")
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

for i, animal in enumerate(os.listdir(data_path)):
    if os.path.isdir(os.path.join(data_path, animal)):
        animal_path = os.path.join(data_path, animal)
        images = [f for f in os.listdir(animal_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            # Load the first image from this animal
            img_path = os.path.join(animal_path, images[0])
            img = Image.open(img_path)
            
            axes[i].imshow(img)
            axes[i].set_title(f"{animal}\n({len(images)} images)")
            axes[i].axis('off')

plt.tight_layout()
plt.show()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Define transforms (how we'll process the images)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Make all images 64x64 pixels
    transforms.ToTensor()         # Convert to numbers (0-1 range)
])

print("Transforms defined:")
print("  - Resize to 64x64 pixels")
print("  - Convert to tensor (numbers)")

class CowDetectorDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Find the cow folder
        cow_folder = os.path.join(data_path, "mucca")
        if not os.path.exists(cow_folder):
            raise ValueError("Cow folder (mucca) not found!")
        
        # Load cow images (label = 0)
        cow_images = [f for f in os.listdir(cow_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in cow_images:
            self.images.append(os.path.join(cow_folder, img))
            self.labels.append(0)  # 0 = cow
        
        # Load non-cow images (label = 1)
        for animal in os.listdir(data_path):
            if animal != "mucca" and os.path.isdir(os.path.join(data_path, animal)):
                animal_path = os.path.join(data_path, animal)
                animal_images = [f for f in os.listdir(animal_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Take a random sample to balance the dataset
                sample_size = min(len(animal_images), 200)  # Max 200 per non-cow animal
                sampled_images = random.sample(animal_images, sample_size)
                
                for img in sampled_images:
                    self.images.append(os.path.join(animal_path, img))
                    self.labels.append(1)  # 1 = not cow
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Create the dataset
print("Creating dataset...")
dataset = CowDetectorDataset(data_path, transform=transform)
print(f"Total images: {len(dataset)}")

# Count cow vs non-cow
cow_count = sum(1 for label in dataset.labels if label == 0)
non_cow_count = sum(1 for label in dataset.labels if label == 1)
print(f"Cow images: {cow_count}")
print(f"Non-cow images: {non_cow_count}")
# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Define transforms (how we'll process the images)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Make all images 64x64 pixels
    transforms.ToTensor()         # Convert to numbers (0-1 range)
])

print("Transforms defined:")
print("  - Resize to 64x64 pixels")
print("  - Convert to tensor (numbers)")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Define transforms (how we'll process the images)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Make all images 64x64 pixels
    transforms.ToTensor()         # Convert to numbers (0-1 range)
])

print("Transforms defined:")
print("  - Resize to 64x64 pixels")
print("  - Convert to tensor (numbers)")

# Create the dataset
print("Creating dataset...")
dataset = CowDetectorDataset(data_path, transform=transform)
print(f"Total images: {len(dataset)}")

# Count cow vs non-cow
cow_count = sum(1 for label in dataset.labels if label == 0)
non_cow_count = sum(1 for label in dataset.labels if label == 1)
print(f"Cow images: {cow_count}")
print(f"Non-cow images: {non_cow_count}")

# Let's look at a few examples
print("\nLet's look at some examples:")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

for i in range(8):
    image, label = dataset[i]
    
    # Convert tensor back to image for display
    image_display = image.permute(1, 2, 0)
    
    axes[i].imshow(image_display)
    axes[i].set_title(f"Label: {label} ({'Cow' if label == 0 else 'Not Cow'})")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Let's look at examples from both categories
print("Let's look at examples from both categories:")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

# Get some cow examples (label 0)
cow_indices = [i for i, label in enumerate(dataset.labels) if label == 0][:4]
# Get some non-cow examples (label 1)
non_cow_indices = [i for i, label in enumerate(dataset.labels) if label == 1][:4]

# Combine them
example_indices = cow_indices + non_cow_indices

for i, idx in enumerate(example_indices):
    image, label = dataset[idx]
    
    # Convert tensor back to image for display
    image_display = image.permute(1, 2, 0)
    
    axes[i].imshow(image_display)
    axes[i].set_title(f"Label: {label} ({'Cow' if label == 0 else 'Not Cow'})")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Let's also check the distribution
print(f"\nDataset distribution:")
print(f"Total images: {len(dataset)}")
print(f"Cow images: {cow_count}")
print(f"Non-cow images: {non_cow_count}")
print(f"Ratio: {cow_count/non_cow_count:.2f}")

# Split the dataset into training and testing
from torch.utils.data import random_split

# 80% for training, 20% for testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Training set: {len(train_dataset)} images")
print(f"Testing set: {len(test_dataset)} images")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Testing batches: {len(test_loader)}")

# Let's see what a batch looks like
print("Let's look at a training batch:")
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label values: {labels[:10]}")  # First 10 labels
    
    # Show the first 8 images in this batch
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(8):
        image = images[i]
        label = labels[i]
        
        # Convert tensor back to image for display
        image_display = image.permute(1, 2, 0)
        
        axes[i].imshow(image_display)
        axes[i].set_title(f"Label: {label.item()} ({'Cow' if label.item() == 0 else 'Not Cow'})")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    break  # Only show the first batch

class CowDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First conv block - looks for basic patterns like edges, shapes
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3 color channels -> 16 features
            nn.ReLU(),                       # Makes the network more powerful
            nn.Conv2d(16, 16, 3, padding=1), # 16 -> 16 features
            nn.ReLU(),
            nn.MaxPool2d(2)                  # Reduce size by half
        )
        
        # Second conv block - looks for more complex patterns
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), # 16 -> 32 features
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), # 32 -> 32 features
            nn.ReLU(),
            nn.MaxPool2d(2)                  # Reduce size by half again
        )
        
        # Final classifier - decides cow or not cow
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # Convert 2D to 1D
            nn.Linear(32*16*16, 2)          # 2 outputs: cow or not cow
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# Create the model
model = CowDetector().to(device)
print("Model created!")
print(f"Model structure:\n{model}")

# Let's understand what happens to an image as it goes through the model
print("Let's trace an image through the model:")

# Get a sample image
sample_image, sample_label = next(iter(train_loader))
sample_image = sample_image[0:1].to(device)  # Take just one image AND move to GPU
sample_label = sample_label[0:1].to(device)  # Move label to GPU too

print(f"Input image shape: {sample_image.shape}")

# Go through each layer
x = sample_image
print(f"After conv_block_1: {model.conv_block_1(x).shape}")
x = model.conv_block_1(x)
print(f"After conv_block_2: {model.conv_block_2(x).shape}")
x = model.conv_block_2(x)
print(f"After classifier: {model.classifier(x).shape}")

# Make a prediction
model.eval()
with torch.no_grad():
    output = model(sample_image)
    prediction = torch.argmax(output, dim=1)
    confidence = torch.softmax(output, dim=1)
    
print(f"\nRaw output: {output}")
print(f"Prediction: {prediction.item()} ({'Cow' if prediction.item() == 0 else 'Not Cow'})")
print(f"Confidence: {confidence[0].tolist()}")
print(f"Actual label: {sample_label.item()} ({'Cow' if sample_label.item() == 0 else 'Not Cow'})")

# Set up training
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer

print("Training setup:")
print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Learning rate: 0.001")

# Let's test the model before training
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

initial_accuracy = 100 * test_correct / test_total
print(f"Initial accuracy (before training): {initial_accuracy:.2f}%")

# Training loop
print("Starting training...")
num_epochs = 10

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    train_accuracy = 100 * train_correct / train_total
    avg_loss = train_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

# Test the final model
print("\nTesting final model...")
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

final_accuracy = 100 * test_correct / test_total
print(f"Final test accuracy: {final_accuracy:.2f}%")