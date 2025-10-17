import torch
torch.manual_seed(0)
torch.cuda.manual_seed(1234)

random_tensor = torch.randn(7, 7, device='cuda')
tensor2 = torch.randn(1, 7, device='cuda')


result = random_tensor @ tensor2.T


torch.manual_seed(1234)
lol1 = torch.randn(2, 3, device='cuda')
lol2 = torch.randn(2, 3, device='cuda')

result2 = torch.mm(lol1, lol2.T)
#print(result2.max())
#print(result2.min())

#print(result2.argmax())
#print(result2.argmin())

torch.manual_seed(7)

l1 = torch.randn(1, 1, 1, 10)

l2 = l1.squeeze()

print(l1, l1.shape)


print(l2, l2.shape)


