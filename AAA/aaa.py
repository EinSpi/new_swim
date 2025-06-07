import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt

# 原始函数（无极点）
def true_func(x):
    return np.sin(2 * np.pi * x) + 0.1 * x

# 给定10个数据点
x_data = np.linspace(0, 1, 10)
y_data = true_func(x_data)

# 支撑点数量上限
max_support = 6
support_indices = []

# 贪婪选择支撑点
for step in range(max_support):
    if step == 0:
        support_indices.append(np.argmax(np.abs(y_data)))
    else:
        xs = x_data[support_indices]
        ys = y_data[support_indices]

        def r(x):
            num = 0
            denom = 0
            for j in range(len(xs)):
                wj = 1.0
                num += wj * ys[j] / (x - xs[j] + 1e-10)
                denom += wj / (x - xs[j] + 1e-10)
            return num / denom

        residual = np.abs(r(x_data) - y_data)
        residual[support_indices] = -1
        support_indices.append(np.argmax(residual))

# 最终拟合函数
xs = x_data[support_indices]
ys = y_data[support_indices]

def r_final(x):
    num = 0
    denom = 0
    for j in range(len(xs)):
        wj = 1.0
        num += wj * ys[j] / (x - xs[j] + 1e-10)
        denom += wj / (x - xs[j] + 1e-10)
    return num / denom

# 拟合结果
x_dense = np.linspace(0, 1, 500)
y_dense = r_final(x_dense)

# 可视化
plt.figure(figsize=(8, 5))
plt.plot(x_dense, y_dense, label='AAA Approximation (6 support)', linewidth=2)
plt.scatter(x_data, y_data, color='red', label='Original Data')
plt.scatter(xs, ys, color='blue', marker='x', s=80, label='Support Points')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Fixed Support AAA Rational Approximation")
plt.grid(True)
plt.tight_layout()
plt.show()
