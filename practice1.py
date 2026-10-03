import numpy as np
import torch

# 选择 x, y
x = 2.0
y = 3.0

# ===== NumPy 前向传播 =====
a = x * y
b = x ** 2
c = a + b
f = c ** 2

print("NumPy 前向：")
print("a =", a)
print("b =", b)
print("c =", c)
print("f =", f)

# ===== NumPy 反向传播 =====
df_dc = 2 * c
dc_da = 1.0
dc_db = 1.0
da_dx = y
da_dy = x
db_dx = 2 * x

df_dx = df_dc * (dc_da * da_dx + dc_db * db_dx)
df_dy = df_dc * (dc_da * da_dy)

print("\nNumPy 手算梯度：")
print("df/dx =", df_dx)
print("df/dy =", df_dy)

# ===== PyTorch autograd 验证 =====
x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
y_t = torch.tensor(y, dtype=torch.float64, requires_grad=True)

a_t = x_t * y_t
b_t = x_t ** 2
c_t = a_t + b_t
f_t = c_t ** 2

f_t.backward()

print("\nPyTorch autograd 梯度：")
print("df/dx =", x_t.grad.item())
print("df/dy =", y_t.grad.item())

print("\n误差：")
print("x 误差 =", abs(df_dx - x_t.grad.item()))
print("y 误差 =", abs(df_dy - y_t.grad.item()))