import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
import matplotlib.pyplot as plt

def ground_truth(N):
    """
    生成一个形状为 (N, 1) 的张量，值由给定函数 func 决定。

    参数:
        N (int): 长度
        func (callable): 一个接受 1D Tensor 并返回 1D Tensor 的函数
    
    返回:
        Tensor: shape (N, 1)
    """
    x = torch.arange(N, dtype=torch.float32)  # 可按需要改为其他坐标值
    y = torch.sin(0.005*x).unsqueeze(1)  # shape: (N, 1)
    return y

def generate_activation_pool(l=0,r=999, N=1000,step=3, func=torch.tanh, scale=1.0):
    """
    生成一个 (N, ceil(N/step)) 矩阵，每列是 func(x - b_i)，b_i 从 1 到 N，间隔为 step。
    
    参数:
        N (int): 行数（每列向量的长度）
        step (int): b 的步长（偏移间隔）
        func (callable): 激活函数，如 torch.tanh
    
    返回:
        Tensor: shape (N, ceil(N / step))
    """
    x = torch.arange(N).unsqueeze(1).float()                # shape: (N, 1)
    b = torch.arange(l, r + 1e-6, step).unsqueeze(0).float()  # shape (1, K)   # shape: (1, ceil(N / step))
    shifted = x - b                                         # shape: (N, ceil(N / step))
    return func(scale*shifted)

def generate_ones(N, l, r):
    """
    生成一个形状为 (N, 1) 的张量，其中在索引区间 [l, r] 范围内为1，其余为0。

    参数:
        N (int): 向量长度
        l (int): 区间左端点（包含）
        r (int): 区间右端点（包含）

    返回:
        Tensor: shape (N, 1)，值为0或1
    """
    x = torch.arange(N).unsqueeze(1)
    mask = ((x >= l) & (x <= r)).float()
    return mask

def gaussian(x):
    return torch.exp(-(x)**2)

def plot(tensor:torch.Tensor):
    ###plotting###
    data = tensor.numpy()
    m, N = data.shape

    # 横坐标
    x = list(range(N))

    #plt.plot(x, data[0], label=f'Row {0}')

    # 每一行画一条线
    for i in range(m):
        plt.plot(x, data[i], label=f'Row {i}')
        #print(data[i,6000])

    plt.xlabel("Column Index")
    plt.ylabel("Value")
    plt.title("Each Row as a Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def p_inv(tensor:torch.Tensor,threshold=1e-5):
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    
    # 设置阈值去掉近似为0的奇异值
    #print(S)

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

    """
    #固定前k个的方法
    num=9
    U_k = U[:, :num]             # shape: (n, k)
    S_k = S[:num]                # shape: (k,)
    V_k = Vh[:num, :].T          # shape: (m, k)
    S_k_inv = torch.diag(1.0 / S_k)

    """

    # 计算稳定的伪逆
    #p_prime = V_k @ S_k_inv @ U_k.T  # Shape: (4, k) @ (k, k) @ (k, 10) => (4, 10)
    p_prime=Vh.T @ torch.diag(1.0/S) @ U.T
    return p_prime

N,step=1000, 2
l,r=0,999
f=ground_truth(N) #(N,1)
pool=generate_activation_pool(N=N,l=l,r=r,step=step,func=gaussian,scale=0.05)#(N,m)
ones= generate_ones(N=N,l=l,r=r)
#pool=torch.cat([pool,ones],dim=1) #(N,m+1)
pool_size=pool.shape[1]
width=pool_size//2#select half of all activations
ps=p_inv(pool)
probability=(ps@f).squeeze(-1) #(m,)
probability=torch.abs(probability)
probability=probability/torch.sum(probability)
selected_indices=torch.multinomial(probability, num_samples=width, replacement=True)
selected_activations=pool[:,selected_indices] #(N,w)
least_square_weights,*_ = torch.linalg.lstsq(selected_activations, f) #(w,1)
pred=selected_activations@least_square_weights #(N,1)
pred_and_gt=torch.cat([pred,f],dim=1) #(N,2)
plot(pred_and_gt.T)




