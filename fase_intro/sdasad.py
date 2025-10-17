# Create tensor (default on CPU)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device

tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)