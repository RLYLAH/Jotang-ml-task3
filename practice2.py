import numpy as np
import torch
from sklearn.datasets import make_moons

np.random.seed(42)
torch.manual_seed(42)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def numpy_forward_backward(X, y, W1, b1, W2, b2):
    """
    X:  (N, 2)
    y:  (N, 1)
    W1: (H, 2)
    b1: (1, H)
    W2: (1, H)
    b2: (1, 1)
    """
    N = X.shape[0]

    # 前向传播
    z1 = X @ W1.T + b1          # (N, H)
    a1 = np.maximum(0.0, z1)    # (N, H)
    z2 = a1 @ W2.T + b2         # (N, 1)
    p = sigmoid(z2)             # (N, 1)

    eps = 1e-12
    loss = -np.mean(
        y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)
    )

    # 反向传播
    dz2 = (p - y) / N           # (N, 1)
    dW2 = dz2.T @ a1            # (1, H)
    db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, 1)

    da1 = dz2 @ W2              # (N, H)
    dz1 = da1 * (z1 > 0.0)      # (N, H)

    dW1 = dz1.T @ X             # (H, 2)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, H)

    return {
        "loss": loss,
        "p": p,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }


def torch_forward_backward(X, y, W1, b1, W2, b2):
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    # 前向传播
    z1_t = X_t @ W1_t.T + b1_t      # (N, H)
    a1_t = torch.relu(z1_t)         # (N, H)
    z2_t = a1_t @ W2_t.T + b2_t     # (N, 1)
    p_t = torch.sigmoid(z2_t)       # (N, 1)

    eps = 1e-12
    loss_t = -torch.mean(
        y_t * torch.log(p_t + eps)
        + (1.0 - y_t) * torch.log(1.0 - p_t + eps)
    )

    loss_t.backward()

    return {
        "loss": loss_t.item(),
        "p": p_t.detach().numpy(),
        "z1": z1_t.detach().numpy(),
        "a1": a1_t.detach().numpy(),
        "z2": z2_t.detach().numpy(),
        "dW1": W1_t.grad.detach().numpy(),
        "db1": b1_t.grad.detach().numpy(),
        "dW2": W2_t.grad.detach().numpy(),
        "db2": b2_t.grad.detach().numpy(),
    }


def compare(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    err = np.max(np.abs(a - b))
    ok = np.allclose(a, b, atol=1e-8, rtol=1e-6)

    print(f"\n{name}")
    print("  NumPy 形状:  ", a.shape)
    print("  PyTorch 形状:", b.shape)
    print("  NumPy 值:\n", a)
    print("  PyTorch 值:\n", b)
    print("  最大绝对误差:", err)
    print("  是否一致:", ok)


# ===== 准备 make_moons 数据 =====
X_all, y_all = make_moons(n_samples=200, noise=0.2, random_state=42)
X_all = X_all.astype(np.float64)
y_all = y_all.astype(np.float64).reshape(-1, 1)

# ===== 初始化参数 =====
D = 2
H = 4

W1 = np.random.randn(H, D) * 0.5   # (H, 2)
b1 = np.zeros((1, H))               # (1, H)
W2 = np.random.randn(1, H) * 0.5    # (1, H)
b2 = np.zeros((1, 1))               # (1, 1)

print("参数形状：")
print("X:  (N, 2)")
print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)
print("中间结果形状：z1=(N,H), a1=(N,H), z2=(N,1), p=(N,1), loss=标量")

# ===== 先单样本，再小 batch =====
for N in [1, 8]:
    print("\n" + "=" * 70)
    print(f"开始验证 N = {N}")

    X = X_all[:N]   # (N, 2)
    y = y_all[:N]   # (N, 1)

    np_out = numpy_forward_backward(X, y, W1, b1, W2, b2)
    torch_out = torch_forward_backward(X, y, W1, b1, W2, b2)

    print("loss NumPy:  ", np_out["loss"])
    print("loss PyTorch:", torch_out["loss"])

    # 比较 p
    compare("p", np_out["p"], torch_out["p"])

    # 比较四个参数的梯度
    compare("dW1", np_out["dW1"], torch_out["dW1"])
    compare("db1", np_out["db1"], torch_out["db1"])
    compare("dW2", np_out["dW2"], torch_out["dW2"])
    compare("db2", np_out["db2"], torch_out["db2"])