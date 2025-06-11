import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
import matplotlib.pyplot as plt


def off_diagonal_ratio(A):
    diag_A = torch.diag(torch.diag(A))
    off = A - diag_A
    ratio = torch.norm(off)**2 / torch.norm(A)**2
    return ratio.item()

def projection_matrix(C:torch.Tensor):
    """
    给定 C ∈ ℝ^{N×m}，返回投影到 Col(C) 的正交投影矩阵 P ∈ ℝ^{N×N}
    """
    CtC_inv = torch.linalg.inv(C.T @ C)  # (m, m)
    P = C @ CtC_inv @ C.T               # (N, N)
    return P

def threshold_svd(tensor:torch.Tensor,threshold=1e-5,num=1):
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    # 设置阈值去掉近似为0的奇异值
    S=S.unsqueeze(0)
    plot(S)

    """
    #阈值法
    thres = threshold * S.max()
    valid = S > thres  # 布尔掩码，保留有效奇异值
    k = valid.sum().item()  # 有效秩
    print(k)
    # 截断 SVD 分量
    U_k = U[:, valid]
    S_k_inv = torch.diag(1.0 / S[valid])
    V_k = Vh[valid, :].T
    """

    #固定前k个的方法
    U_k = U[:, :num]             # shape: (n, k)
    S_k = S[:num]                # shape: (k,)
    V_k = Vh[:num, :].T          # shape: (m, k)
    S_k_inv = torch.diag(1.0 / S_k)



    # 计算稳定的伪逆
    p_prime = V_k @ S_k_inv @ U_k.T  # Shape: (4, k) @ (k, k) @ (k, 10) => (4, 10)
    return p_prime

def tanh_tensor(zero_points:torch.Tensor,scale:float=50.0,length=10000):
    # 构造列坐标 0~999，形状为 (1, 1000)
    x = torch.arange(length).unsqueeze(0)  # shape: (1, 1000)

    # 广播生成偏移后的 tanh 输入，每行减去自己的零点位置
    # 结果 shape: (6, 1000)
    shifted_x = x - zero_points.unsqueeze(1)

    # 应用 tanh
    tanh_tensor = torch.tanh(shifted_x *scale)  # /50控制斜率平缓程度，可调
    return tanh_tensor

def impulse_generator(a, length=10000):
    a = a.long()
    m = a.shape[0]
    b = torch.zeros(m, length)
    
    # 检查 a[i] + 1 是否越界
    if torch.any(a >= length - 1):
        raise ValueError("a[i] + 1 exceeds vector length")

    b[torch.arange(m), a] = -1.0
    b[torch.arange(m), a + 1] = 1.0
    return b

def collapse_to_basis_svd(C, num=1):
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    U_r = U[:, :num]           # shape: (N, r)
    S_r = S[:num]              # shape: (r,)
    Vh_r = Vh[:num, :]         # shape: (r, m)
    return U_r @ torch.diag(S_r)  # shape: (N, r)

def plot(tensor:torch.Tensor):
    ###plotting###
    data = tensor.numpy()
    m, N = data.shape

    # 横坐标
    x = list(range(N))

    #plt.plot(x, data[0], label=f'Row {0}')

    # 每一行画一条线
    for i in range(m):
        plt.plot(x[0:20], data[i][0:20], label=f'Row {i}')
        #print(data[i,6000])

    plt.xlabel("Column Index")
    plt.ylabel("Value")
    plt.title("Each Row as a Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

critical_points=torch.arange(0, 5000, step=2 ,dtype=torch.float32)
tanh=tanh_tensor(zero_points=critical_points,scale=10).T #(N,m)
impulse=impulse_generator(a=critical_points).T #(N,m)

_,m=tanh.shape
C=threshold_svd(tensor=tanh,num=m//2).T#(N,m)
C_truncated=collapse_to_basis_svd(C,num=m//2)#(N,m//2)
tanh_truncated=collapse_to_basis_svd(tanh,num=m//2)#(N,m//2)
#C=impulse
#验证对角性
P=projection_matrix(C_truncated)
diag_truncated=tanh_truncated.T@P@tanh_truncated
print(off_diagonal_ratio(diag_truncated))

#计算prefix,除了P,用满血版
diag=tanh.T@P@tanh
prefix_f=torch.linalg.inv(diag)@tanh.T@P
print(prefix_f[:,:10])
plot(prefix_f)
