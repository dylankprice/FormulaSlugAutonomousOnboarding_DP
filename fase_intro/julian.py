import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

#data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(1) # creates data range puts into vector
y = weight * X + bias # y = weight * input + bias

#Split data between test and train
train_split = int(0.8 * len(X)) # 80% of data used for training set
X_train, y_train = X[:train_split], y[:train_split]# takes first 80% of data, X inputs and y outputs
X_test, y_test = X[train_split:], y[train_split:] # takes last 20% of data, X inputs and y outputs

class LinearRegressionModelIV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)

model_1 = LinearRegressionModelIV2()

       


def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});

plot_predictions(X_train, y_train, X_test, y_test)


model_1.to(device)

loss_fn = nn.L1Loss() #loss function

optimizer = torch.optim.SGD(params=model_1.parameters(),#optimizes models parameters
                            lr=0.01)#learning rate

epochs = 1000

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
   model_1.train() #train

   y_pred = model_1(X_train) # forward pass

   loss = loss_fn(y_pred, y_train)#calculate loss 

   optimizer.zero_grad()#resets optimizer to zero

   loss.backward()#backpropagation(finds gradient)

   optimizer.step()#adjusts parameters based on gradients

   model_1.eval()

   with torch.inference_mode(): 
         test_pred = model_1(X_test) # forward pass

         test_loss = loss_fn(test_pred, y_test)#calculate loss

   #if epoch % 100 == 0:
       #print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)
y_preds

plot_predictions(predictions=y_preds.cpu())

from pathlib import Path
plt.show()

# Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dictionary
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)

# Check the saved file path