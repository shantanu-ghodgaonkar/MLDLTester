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


tensor = torch.tensor([1, 2, 3])
print(f"Tensor = {tensor}")
# print(f"Tensor + 10 = {tensor + 10}")
# print(f"Tensor * 10 = {tensor * 10}")
# print(f"Tensor / 10 = {tensor / 10}")
# print(f"Tensor - 10 = {tensor - 10}")
# print(f"Tensor * 10 = {torch.mul(tensor, 10)}")
print(f"tensor * tensor = {tensor * tensor}")
print(f"tensor matmul tensor = {torch.matmul(tensor, tensor)}")