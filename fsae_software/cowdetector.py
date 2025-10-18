import torch
from torch import nn
from torchvision import transforms
import os
from PIL import Image
import random
import kagglehub

#This code trains a simple neural network to classify images as cows or not-cows (dogs and cats) using a dataset from Kaggle

#There is a seperate file with results so you dont need to run if you dont want to set it up

path = kagglehub.dataset_download("alessiocorrado99/animals10")
data_path = os.path.join(path, "raw-img")  # Navigate to the raw-img subfolder

device = "cuda" if torch.cuda.is_available() else "cpu"

#makes images all 32x32 and turns them from images to tensors
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


X = []  # image data
y = []  # output labels (0 for cow, 1 for not-cow)

# Load cow images (label = 0)
cow_folder = os.path.join(data_path, "mucca")
cow_images = [f for f in os.listdir(cow_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for img in cow_images[:500]:
    image = Image.open(os.path.join(cow_folder, img)).convert('RGB')
    image = transform(image)
    X.append(image)
    y.append(0)

# Load non-cow images (label = 1) - dogs and cats
dog_folder = os.path.join(data_path, "cane")
dog_images = [f for f in os.listdir(dog_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for img in dog_images[:250]:  # 250 dogs
    image = Image.open(os.path.join(dog_folder, img)).convert('RGB')
    image = transform(image)
    X.append(image)
    y.append(1)

cat_folder = os.path.join(data_path, "gatto")
cat_images = [f for f in os.listdir(cat_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for img in cat_images[:250]:  # 250 cats
    image = Image.open(os.path.join(cat_folder, img)).convert('RGB')
    image = transform(image)
    X.append(image)
    y.append(1)

# convert to tensors
X = torch.stack(X).to(device)
y = torch.tensor(y).to(device)

# creates balanced split of cows and not-cows
cow_indices = [i for i, label in enumerate(y) if label == 0] # list of indices for cow
not_cow_indices = [i for i, label in enumerate(y) if label == 1] # list of indices for not-cow

# data split 80/20 for cows
cow_train_size = int(0.8 * len(cow_indices))
cow_train_indices = cow_indices[:cow_train_size]
cow_test_indices = cow_indices[cow_train_size:]

# data split 80/20 for not-cows
not_cow_train_size = int(0.8 * len(not_cow_indices))
not_cow_train_indices = not_cow_indices[:not_cow_train_size]
not_cow_test_indices = not_cow_indices[not_cow_train_size:]

# Combine train and test indices
train_indices = cow_train_indices + not_cow_train_indices
test_indices = cow_test_indices + not_cow_test_indices

# Create train and test sets
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

print(f"Training set: {len(X_train)} images ({sum(y_train == 0)} cows, {sum(y_train == 1)} not-cows)")
print(f"Test set: {len(X_test)} images ({sum(y_test == 0)} cows, {sum(y_test == 1)} not-cows)")


class CowDetectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(32*32*3, 128) # 3 color channels of 32 x 32 images, 128 hidden layer size(number of features the model can learn)
        self.linear_layer2 = nn.Linear(128, 2)#2 outputs: cow(0) or not-cow(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)  #removes extra dimensions/ same amount of elemnts but flattens each image to a vector
        x = self.linear_layer(x)
        x = self.linear_layer2(x)
        return x

torch.manual_seed(67)#six-seven seed

model_2 = CowDetectorModel()#assign CowDetectorModel to model_2 and cuda
model_2.to(device)

loss_fn = nn.CrossEntropyLoss() # Classification loss function
optimizer = torch.optim.SGD(params=model_2.parameters(), 
                            lr=0.01)
epochs = 100

# Training loop
for epoch in range(epochs):
    model_2.train()
    y_pred = model_2(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model_2.eval()
with torch.inference_mode():
    y_preds = model_2(X_test)

# Find some cow examples and some not-cow examples
cow_indices = [i for i, label in enumerate(y_test) if label == 0][:10]  # First 10 cows
not_cow_indices = [i for i, label in enumerate(y_test) if label == 1][:10]  # First 10 not-cows

#Shows examples with cow images and not-cow images
print("COW EXAMPLES:")
for i, index in enumerate(cow_indices):
    predicted_class = torch.argmax(y_preds[index]).item()
    actual_class = y_test[index].item()
    print(f"Cow Example {i+1}: Predicted = {predicted_class} ({'Cow' if predicted_class == 0 else 'Not Cow'}), "
          f"Actual = {actual_class} ({'Cow' if actual_class == 0 else 'Not Cow'})")

print("\nNOT-COW EXAMPLES:")
for i, index in enumerate(not_cow_indices):
    predicted_class = torch.argmax(y_preds[index]).item()
    actual_class = y_test[index].item()
    print(f"Not-Cow Example {i+1}: Predicted = {predicted_class} ({'Cow' if predicted_class == 0 else 'Not Cow'}), "
          f"Actual = {actual_class} ({'Cow' if actual_class == 0 else 'Not Cow'})")

# Calculate accuracy
correct = (torch.argmax(y_preds, dim=1) == y_test).sum().item()
total = len(y_test)
accuracy = 100 * correct / total
print(f"\nFinal accuracy: {accuracy:.2f}%")

#returns index(0 or 1) of cow or not cow likeliness, based off which is higher
cow_predictions = (torch.argmax(y_preds, dim=1) == 0).sum().item() # true = 1, false = 0, sum adds them up indicating number of cows predicted
not_cow_predictions = (torch.argmax(y_preds, dim=1) == 1).sum().item()
actual_cows = (y_test == 0).sum().item()
actual_not_cows = (y_test == 1).sum().item()

print(f"\nPrediction Summary:")
print(f"Model predicted {cow_predictions} cows and {not_cow_predictions} not-cows")
print(f"Actually there were {actual_cows} cows and {actual_not_cows} not-cows")