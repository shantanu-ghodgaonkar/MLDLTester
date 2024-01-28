import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(torch.__version__)

# scalar = torch.tensor(7)
# print(scalar)
# print(scalar.ndim)
# print(scalar.item())

# vector = torch.tensor([7, 7])
# print(vector)
# print(vector.ndim)
# print(vector.shape)

# MATRIX = torch.tensor([[7, 8],
#                        [9, 10]])
# print(MATRIX)

# print(MATRIX.ndim)
# print(MATRIX.shape)
# print(MATRIX[0])
# print(MATRIX[1])

# TENSOR = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 5, 4]],[[4, 5, 6], [3, 5, 6], [1, 6, 9]]])
# print(TENSOR)
# print(TENSOR.ndim)
# print(TENSOR.shape)
# print(TENSOR[0][1][1])

# RANDOM TENSORS

# random_tensor = torch.rand(3, 4)
# print(random_tensor)
# print(random_tensor.ndim)
# print(random_tensor.shape)

# random_image_size_tensor = torch.rand(size=(3, 224, 224))
# print(random_image_size_tensor.shape)
# print(random_image_size_tensor.ndim)

# zeros = torch.zeros(3,4)
# ones = torch.ones(3,4)
# print(random_tensor*zeros)

# one_to_ten = torch.arange(start=1, end=11, step=1)
# print(one_to_ten)
# ten_zeros = torch.zeros_like(input=one_to_ten)
# print(ten_zeros)

# float_32_tensor = torch.rand(size=(3,4), dtype=torch.float32, device="cuda", requires_grad=False)
# float_32_tensor = torch.rand(size=(3,4), dtype=torch.float32, device=None, requires_grad=False)
# # print(float_32_tensor.dtype)

# float_16_tensor = float_32_tensor.type(torch.float16)
# # print(float_16_tensor.dtype)

# # print((float_16_tensor*float_32_tensor).dtype)

# int_32_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32)

# # print(int_32_tensor.dtype)

# print(float_16_tensor*int_32_tensor)


# some_tensor = torch.rand(size=(3,4), device="cuda")
# print(some_tensor)
# print(f"Datatype: {some_tensor.dtype}")
# print(f"Shape: {some_tensor.shape}")
# print(f"Device: {some_tensor.device}")
# print(f"Size: {some_tensor.size()}")


# tensor = torch.tensor([1, 2, 3])
# print(f"Tensor = {tensor}")
# # print(f"Tensor + 10 = {tensor + 10}")
# # print(f"Tensor * 10 = {tensor * 10}")
# # print(f"Tensor / 10 = {tensor / 10}")
# # print(f"Tensor - 10 = {tensor - 10}")
# # print(f"Tensor * 10 = {torch.mul(tensor, 10)}")
# print(f"tensor * tensor = {tensor * tensor}")
# print(f"tensor matmul tensor = {torch.matmul(tensor, tensor)}")

# tensor_A = torch.rand(3,2)
# tensor_B = torch.rand(3,2)
# print(torch.mm(tensor_A.T, tensor_B))

# x = torch.arange(0, 100, 10)

# print(x)
# # print(f"Min = {x.min()}")
# # print(f"Max = {x.max()}")
# # print(f"Mean = {x.type(torch.float32).mean()}")
# # print(f"Median = {x.type(torch.float32).median()}")
# # print(f"Median = {x.type(torch.float32).mode()}")
# # print(f"Sum of Elements = {x.sum()}")

# print(f"Index location of x.min() value = {x.argmin()}")
# print(f"Index location of x.max() value = {x.argmax()}")

# x = torch.arange(1., 10.)
# print(f"x = {x}")
# print(f"x.shape = {x.shape}")

# x_reshaped = x.reshape(3,1,3)
# print(f"x_reshaped = {x_reshaped}")
# print(f"x_reshaped.shape = {x_reshaped.shape}")

# z = x.view(3,3)
# print(f"z = {z}")
# print(f"z.shape = {z.shape}")

# NOTE: Changing z changes x because a view of a tensor shares the same memory location as the original tensor

# x_stacked = torch.stack([x,x,x,x],dim=0)
# print(f"x_stacked = {x_stacked}")
# print(f"x_stacked.shape = {x_stacked.shape}")

# NOTE: See torch.vstack and torch.hstack

# x_squeeze = x_reshaped.squeeze()
# print(f"x_squeeze = {x_squeeze}")
# print(f"x_squeeze.shape = {x_squeeze.shape}")

# x_unsqueeze = x_squeeze.unsqueeze(dim=0)
# print(f"x_unsqueeze = {x_unsqueeze}")
# print(f"x_unsqueeze.shape = {x_unsqueeze.shape}")

# x = torch.rand(size=(224,224,3)) # [height, width, colour channel]
# print(f"x = {x}")
# print(f"x.shape = {x.shape}")

# x_permute = x.permute(2,0,1)
# print(f"x_permute = {x_permute}") # [colour channel, height, width]
# print(f"x_permute.shape = {x_permute.shape}")
# # NOTE: torch.permute also returns a view, so be careful


