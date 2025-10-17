import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class AdditionNet(nn.Module):
    def __init__(self):
        super(AdditionNet, self).__init__()
        # Simple neural network: 2 input -> 10 hidden -> 10 hidden -> 1 output
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output layer
        return x

def generate_data(num_samples=1000):
    """Generate training data for addition"""
    # Create random pairs of numbers between 0 and 100
    X = torch.randint(0, 101, (num_samples, 2), dtype=torch.float32)
    
    # Calculate sums
    y = X.sum(dim=1, keepdim=True)  # Sum along dimension 1, keep as column vector
    
    return X, y

def train_model(model, X_train, y_train, epochs=100):
    """Train the neural network"""
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    
    model.train()  # Set to training mode
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def test_model(model, test_cases):
    """Test the trained model"""
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():  # Don't track gradients for testing
        for i, (a, b) in enumerate(test_cases):
            input_tensor = torch.tensor([[a, b]], dtype=torch.float32)
            prediction = model(input_tensor)
            actual = a + b
            
            print(f'Test {i+1}: {a} + {b} = {prediction.item():.2f} (actual: {actual})')

def main():
    print("Creating Addition Neural Network with PyTorch!")
    print("=" * 50)
    
    # Generate training data
    print("Generating training data...")
    X_train, y_train = generate_data(num_samples=2000)
    print(f"Generated {len(X_train)} training samples")
    
    # Create model
    print("\nCreating neural network...")
    model = AdditionNet()
    print(model)
    
    # Train model
    print("\nTraining model...")
    trained_model = train_model(model, X_train, y_train, epochs=200)
    
    # Test the model
    print("\nTesting the trained model:")
    print("-" * 30)
    test_cases = [(5, 3), (10, 15), (50, 25), (99, 1), (7, 8)]
    test_model(trained_model, test_cases)
    
    # Test with some random examples
    print("\nRandom test cases:")
    print("-" * 20)
    random_tests = torch.randint(0, 101, (5, 2), dtype=torch.float32)
    for i, test_input in enumerate(random_tests):
        a, b = test_input[0].item(), test_input[1].item()
        with torch.no_grad():
            prediction = trained_model(test_input.unsqueeze(0))
            print(f'Random {i+1}: {a:.0f} + {b:.0f} = {prediction.item():.2f} (actual: {a+b:.0f})')

if __name__ == "__main__":
    main()