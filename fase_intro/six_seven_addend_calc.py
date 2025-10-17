import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"

X=torch.randint(0,6, (100, 1)).float().to(device) #creates data range of int pair
y =6.7 - X[:, 0]#creates output by adding the pairs

train_split = int(0.8 * len(X)) # 80% of data used for training set
X_train, y_train = X[:train_split], y[:train_split]# takes first 80% of data, X inputs and y outputs
X_test, y_test = X[train_split:], y[train_split:] # takes last 20% of data, X inputs and y outputs

y_train = y_train.unsqueeze(1).float().to(device)#reshapes y to be a column vector and makes it float
y_test = y_test.unsqueeze(1).float().to(device)#reshapes y to be a column vector and makes it float



class LinearRegressionModelIV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)#one input feature(one number) and one output feature(one number)
        self.linear_layer2 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(x)
        x = self.linear_layer2(x)
        return x
    
torch.manual_seed(42)

model_2 = LinearRegressionModelIV2()

model_2.to(device)#sets model to cuda

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_2.parameters(), 
                            lr=0.01)

epochs = 1000

#puts data to cuda
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
   model_2.train() #train

   y_pred = model_2(X_train) # forward pass

   loss = loss_fn(y_pred, y_train)#calculate loss 

   optimizer.zero_grad()#resets optimizer to zero

   loss.backward()#backpropagation(finds gradient)

   optimizer.step()#adjusts parameters based on gradients

   model_2.eval()

   with torch.inference_mode(): 
         test_pred = model_2(X_test) # forward pass

         test_loss = loss_fn(test_pred, y_test)#calculate loss

model_2.eval()#puts model in eval mode

with torch.inference_mode():
    y_preds = model_2(X_test)

for i in range(20):
    print(f"if you add {int(X_test[i])} and {float(y_preds[i]):.2f} you get approximately 6.7 !!")