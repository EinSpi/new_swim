import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from torchmin import minimize

# 1. 给定数据点
x_points = torch.tensor([0.0, 0.5, 1.0, 1.7, 2.0])
y_targets = torch.tensor([1.0, 4.0, 1.5, 6.0, -2])

# 2. 定义有理函数
def rational_function(x, params):
    p0, p1, p2, p3, q1, q2 = params
    numerator = p0 + p1 * x + p2 * x**2 + p3 * x**3
    denominator = q1 * (x+10) * q2 * (x-13) + 1e-8  # 避免除以零
    return numerator / denominator

# 3. 定义损失函数（1 - 余弦相似度）
def loss(params):
    preds = rational_function(x_points, params)
    dot_product = torch.dot(preds, y_targets)
    norm_preds = torch.norm(preds)
    norm_targets = torch.norm(y_targets)
    cosine_similarity = dot_product / (norm_preds * norm_targets + 1e-8)
    return 1.0 - cosine_similarity

# 4. 定义正则项
def regularization(params):
    x_dense = torch.linspace(torch.min(x_points) - 0.5, torch.max(x_points) + 0.5, 200)
    y_dense = rational_function(x_dense, params)
    return torch.mean(y_dense**2)

# 5. 总损失函数
def total_loss(params):
    return loss(params) + 0*regularization(params)

# 6. 初始化参数
init_params = torch.randn(6, dtype=torch.float64, requires_grad=True)

# 7. 优化
result = minimize(total_loss, init_params, method='l-bfgs', tol=1e-9, max_iter=500)
optimal_params = result.x.detach()
print("Optimized parameters:", optimal_params)

# 8. 计算最终余弦相似度
preds_final = rational_function(x_points, optimal_params)
dot_product = torch.dot(preds_final, y_targets)
norm_preds = torch.norm(preds_final)
norm_targets = torch.norm(y_targets)
cosine_similarity_final = dot_product / (norm_preds * norm_targets + 1e-8)
print(f"Final cosine similarity: {cosine_similarity_final.item():.6f}")

# 9. 绘制拟合结果
x_fine = torch.linspace(torch.min(x_points) - 0.2, torch.max(x_points) + 0.2, 400)
y_fine = rational_function(x_fine, optimal_params)

# 计算缩放系数
abs_sum = torch.sum(torch.abs(y_fine))
alpha = 400 / (abs_sum + 1e-8)
y_fitted = alpha * y_fine
y_fitted_at_points = rational_function(x_points, optimal_params)

# 10. 绘图

plt.figure(figsize=(8, 6))
plt.plot(x_fine.cpu(), y_fitted.detach().cpu(), label='Fitted Rational Function', linewidth=2)
plt.scatter(x_points.cpu(), y_targets.cpu(), color='red', label='Given Points', s=50, zorder=5)
plt.scatter(x_points.cpu(), y_fitted_at_points.detach().cpu(), color='blue', marker='x', label='Fitted Points', s=50, zorder=5)
plt.title('Rational Function Fitting by Maximizing Cosine Similarity')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('fitted_result.png') 
#plt.show()




"""import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. 给定 5 个点
x_points = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_targets = np.array([1.0, 4.0, 1.5, 1.0, 0.5])

# 2. 定义 (3,2) 阶的有理函数
def rational_function(x, params):
    p0, p1, p2, p3, q1, q2 = params
    numerator = p0 + p1*x + p2*x**2 + p3*x**3
    denominator = 1 + q1*x + q2*x**2
    return numerator / (denominator + 1e-8)  # 加小量避免除零

# 3. 定义优化的目标：1 - cosine similarity
def loss(params):
    preds = rational_function(x_points, params)
    dot_product = np.dot(preds, y_targets)
    norm_preds = np.linalg.norm(preds)
    norm_targets = np.linalg.norm(y_targets)
    cosine_similarity = dot_product / (norm_preds * norm_targets + 1e-8)
    return 1.0 - cosine_similarity

def regularization(params):
    x_dense = np.linspace(np.min(x_points)-0.5, np.max(x_points)+0.5, 200)  # 密集取点
    y_dense = rational_function(x_dense, params)
    return np.mean(y_dense**2)  # 惩罚整体幅值变大

def total_loss(params):
    cosine_loss = loss(params)
    reg_term = regularization(params)
    return cosine_loss +   reg_term

def scaled_rational_function(alpha,x,optimal_params):
    return alpha * rational_function(x, optimal_params)

# 4. 初始猜测（可以小一点）
init_params = np.random.randn(6) * 0.1


# 5. 优化
result = minimize(total_loss, init_params, method='L-BFGS-B')
optimal_params = result.x
print("Optimized parameters:", optimal_params)


#打cos
preds_final = rational_function(x_points, optimal_params)
dot_product = np.dot(preds_final, y_targets)
norm_preds = np.linalg.norm(preds_final)
norm_targets = np.linalg.norm(y_targets)
cosine_similarity_final = dot_product / (norm_preds * norm_targets + 1e-8)

print(f"Optimization finished. Final cosine similarity = {cosine_similarity_final:.6f}")

# 6. 拟合结果
x_fine = np.linspace(min(x_points)-0.2, max(x_points)+0.2, 400)
y_fine = rational_function(x_fine, optimal_params)
# 计算当前绝对值和
abs_sum = np.sum(np.abs(y_fine))
# 计算放缩系数
alpha = 400 / (abs_sum + 1e-8)

y_fitted = scaled_rational_function(alpha, x_fine, optimal_params)
y_fitted_at_points = rational_function(x_points, optimal_params)

# 7. 画图
plt.figure(figsize=(8, 6))
plt.plot(x_fine, y_fitted, label='Fitted Rational Function', linewidth=2)
plt.scatter(x_points, y_targets, color='red', label='Given Points', s=50, zorder=5)
plt.scatter(x_points, y_fitted_at_points, color='blue', marker='x', label='Fitted Points', s=50, zorder=5)
plt.title('Rational Function Fitting by Maximizing Cosine Similarity')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
"""