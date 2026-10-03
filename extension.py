import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ========== 增加这两行，解决中文方框警告 ==========
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# ================================================

# 让结果可复现
np.random.seed(42)
torch.manual_seed(42)


# =========================================================
# 1. 准备数据
# =========================================================
def load_data(n_samples=1000, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.float64).reshape(-1, 1)

    # 标准化，让训练更稳
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    # 划分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test, X_mean, X_std


# =========================================================
# 2. NumPy 版本：前向 + 反向 + SGD
# =========================================================
def sigmoid(z):
    # 需要数值稳定 sigmoid
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def numpy_forward(X, y, W1, b1, W2, b2):
    """
    X:  (N, 2)
    y:  (N, 1)
    W1: (H, 2)
    b1: (1, H)
    W2: (1, H)
    b2: (1, 1)
    """
    z1 = X @ W1.T + b1          # (N, H)
    a1 = np.maximum(0.0, z1)    # (N, H)
    z2 = a1 @ W2.T + b2         # (N, 1)
    p = sigmoid(z2)             # (N, 1)

    eps = 1e-12
    loss = -np.mean(
        y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)
    )

    cache = (X, y, z1, a1, z2, p)
    return loss, p, cache


def numpy_backward(cache, W2):
    X, y, z1, a1, z2, p = cache
    N = X.shape[0]

    dz2 = (p - y) / N           # (N, 1)
    dW2 = dz2.T @ a1            # (1, H)
    db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, 1)

    da1 = dz2 @ W2              # (N, H)
    dz1 = da1 * (z1 > 0.0)      # (N, H)

    dW1 = dz1.T @ X             # (H, 2)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, H)

    return dW1, db1, dW2, db2


def numpy_predict(X, W1, b1, W2, b2):
    z1 = X @ W1.T + b1
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ W2.T + b2
    p = sigmoid(z2)
    return (p >= 0.5).astype(np.float64)


def numpy_train(X_train, y_train, X_test, y_test,
                hidden=16, lr=0.5, epochs=2000, batch_size=64):
    N, D = X_train.shape
    H = hidden

    # 初始化参数
    W1 = np.random.randn(H, D) * 0.5
    b1 = np.zeros((1, H))
    W2 = np.random.randn(1, H) * 0.5
    b2 = np.zeros((1, 1))

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        # 打乱数据
        idx = np.random.permutation(N)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        # 小批量 SGD
        for start in range(0, N, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]

            loss, p, cache = numpy_forward(Xb, yb, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = numpy_backward(cache, W2)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # 每个 epoch 记录 loss 和 accuracy
        tr_loss, _, _ = numpy_forward(X_train, y_train, W1, b1, W2, b2)
        te_loss, _, _ = numpy_forward(X_test, y_test, W1, b1, W2, b2)

        tr_pred = numpy_predict(X_train, W1, b1, W2, b2)
        te_pred = numpy_predict(X_test, W1, b1, W2, b2)

        tr_acc = (tr_pred == y_train).mean()
        te_acc = (te_pred == y_test).mean()

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        if (epoch + 1) % 200 == 0:
            print(f"[NumPy] epoch {epoch+1:4d} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"test loss {te_loss:.4f} acc {te_acc:.4f}")

    return {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2,
        "train_losses": train_losses, "test_losses": test_losses,
        "train_accs": train_accs, "test_accs": test_accs,
    }


# =========================================================
# 3. PyTorch 版本：同结构网络
# =========================================================
class TorchNet(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # 注意：这里返回 logit，loss 用 BCEWithLogitsLoss
        return x


def torch_train(X_train, y_train, X_test, y_test,
                hidden=16, lr=0.5, epochs=2000, batch_size=64):
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    model = TorchNet(hidden=hidden)
    # 用 SGD，不带 momentum，和 NumPy 版本保持一致
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    N = Xtr.shape[0]
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        idx = torch.randperm(N)
        X_shuf = Xtr[idx]
        y_shuf = ytr[idx]

        for start in range(0, N, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits_tr = model(Xtr)
            logits_te = model(Xte)

            tr_loss = criterion(logits_tr, ytr).item()
            te_loss = criterion(logits_te, yte).item()

            tr_pred = (torch.sigmoid(logits_tr) >= 0.5).float()
            te_pred = (torch.sigmoid(logits_te) >= 0.5).float()

            tr_acc = (tr_pred == ytr).float().mean().item()
            te_acc = (te_pred == yte).float().mean().item()

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        if (epoch + 1) % 200 == 0:
            print(f"[Torch] epoch {epoch+1:4d} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"test loss {te_loss:.4f} acc {te_acc:.4f}")

    return {
        "model": model,
        "train_losses": train_losses, "test_losses": test_losses,
        "train_accs": train_accs, "test_accs": test_accs,
    }


# =========================================================
# 4. 画决策边界
# =========================================================
def plot_decision_boundary(ax, predict_fn, X, y, title):
    # 生成网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict_fn(grid).reshape(xx.shape)

    ax.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(),
               cmap="coolwarm", edgecolors="k", s=20)
    ax.set_title(title)


# =========================================================
# 5. 主流程
# =========================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_mean, X_std = load_data()

    print("数据形状：")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    # ---------- NumPy 训练 ----------
    print("\n===== NumPy 训练 =====")
    np_result = numpy_train(
        X_train, y_train, X_test, y_test,
        hidden=16, lr=0.5, epochs=2000, batch_size=64,
    )

    # ---------- PyTorch 训练 ----------
    print("\n===== PyTorch 训练 =====")
    torch_result = torch_train(
        X_train, y_train, X_test, y_test,
        hidden=16, lr=0.5, epochs=2000, batch_size=64,
    )

    # ---------- 画图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # loss 曲线
    axes[0, 0].plot(np_result["train_losses"], label="NumPy train")
    axes[0, 0].plot(np_result["test_losses"], label="NumPy test", linestyle="--")
    axes[0, 0].plot(torch_result["train_losses"], label="Torch train")
    axes[0, 0].plot(torch_result["test_losses"], label="Torch test", linestyle="--")
    axes[0, 0].set_title("Loss 曲线")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # accuracy 曲线
    axes[0, 1].plot(np_result["train_accs"], label="NumPy train")
    axes[0, 1].plot(np_result["test_accs"], label="NumPy test", linestyle="--")
    axes[0, 1].plot(torch_result["train_accs"], label="Torch train")
    axes[0, 1].plot(torch_result["test_accs"], label="Torch test", linestyle="--")
    axes[0, 1].set_title("Accuracy 曲线")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # NumPy 决策边界
    def numpy_predict_fn(X):
        return numpy_predict(
            X,
            np_result["W1"], np_result["b1"],
            np_result["W2"], np_result["b2"],
        )

    plot_decision_boundary(
        axes[1, 0], numpy_predict_fn,
        X_train, y_train, "NumPy 决策边界（训练集）",
    )

    # PyTorch 决策边界
    def torch_predict_fn(X):
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            logits = torch_result["model"](Xt)
            p = torch.sigmoid(logits).numpy()
        return (p >= 0.5).astype(np.float64)

    plot_decision_boundary(
        axes[1, 1], torch_predict_fn,
        X_train, y_train, "PyTorch 决策边界（训练集）",
    )

    plt.tight_layout()
    plt.savefig("output/extension_result.png", dpi=150)
    plt.show()

    # ---------- 最终对比 ----------
    print("\n===== 最终结果对比 =====")
    print("NumPy  train acc: {:.4f} | test acc: {:.4f}".format(
        np_result["train_accs"][-1], np_result["test_accs"][-1]))
    print("Torch  train acc: {:.4f} | test acc: {:.4f}".format(
        torch_result["train_accs"][-1], torch_result["test_accs"][-1]))