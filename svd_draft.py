import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
import matplotlib.pyplot as plt

def probability_calculator(inte_pool_forward:torch.Tensor,y:torch.Tensor,full_matrices:bool=False)->torch.Tensor:
    U, S, Vh = torch.linalg.svd(inte_pool_forward, full_matrices=False)
    S_inv=torch.diag(1.0 / S)  # (m+1, m+1)
    p = Vh @ S_inv @ U.T @ y # (m+1, 1)
    p=p.squeeze(-1)#(m+1,)
    p=torch.abs(p[:-1])#(m,)
    prob = p / p.sum() #(m,)
    return prob 

tensor = torch.tensor([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0.01, 0.001, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0.01, 0, 0, 0.01, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0.001, 1 ,0],
    [0, 0, 0.02, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0.01, 0, 0, 0, 0, 1, 0.01]
], dtype=torch.float32)

def tanh_tensor(zero_points:torch.Tensor,scale:float=50.0,length=10000):
    # 构造列坐标 0~999，形状为 (1, 1000)
    x = torch.arange(length).unsqueeze(0)  # shape: (1, 1000)

    # 广播生成偏移后的 tanh 输入，每行减去自己的零点位置
    # 结果 shape: (6, 1000)
    shifted_x = x - zero_points.unsqueeze(1)

    # 应用 tanh
    tanh_tensor = torch.tanh(shifted_x / scale)  # /50控制斜率平缓程度，可调
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





def threshold_svd(tensor:torch.Tensor,threshold=1e-5):
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    # 设置阈值去掉近似为0的奇异值
    print(S)

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
    num=9
    U_k = U[:, :num]             # shape: (n, k)
    S_k = S[:num]                # shape: (k,)
    V_k = Vh[:num, :].T          # shape: (m, k)
    S_k_inv = torch.diag(1.0 / S_k)



    # 计算稳定的伪逆
    p_prime = V_k @ S_k_inv @ U_k.T  # Shape: (4, k) @ (k, k) @ (k, 10) => (4, 10)
    return p_prime

def svd(tensor:torch.Tensor):
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    S_inv=torch.diag(1.0 / S)  # (m+1, m+1)
    p =  Vh @ S_inv @ U.T # Shape: (4, k) @ (k, k) @ (k, 10) => (4, 10)
    return p

def plot(tensor:torch.Tensor):
    ###plotting###
    data = tensor.numpy()
    m, N = data.shape

    # 横坐标
    x = list(range(N))

    #plt.plot(x, data[0], label=f'Row {0}')

    # 每一行画一条线
    for i in range(m):
        plt.plot(x[3450:3600], data[i][3450:3600], label=f'Row {i}')
        #print(data[i,6000])

    plt.xlabel("Column Index")
    plt.ylabel("Value")
    plt.title("Each Row as a Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def off_diagonal_ratio(A):
    diag_A = torch.diag(torch.diag(A))
    off = A - diag_A
    ratio = torch.norm(off)**2 / torch.norm(A)**2
    return ratio.item()

#critical_points=torch.tensor([ 3500,3502,3504,3506,3508,3510,3512], dtype=torch.float32)
critical_points=torch.arange(3500, 3550, step=2 ,dtype=torch.float32)
tanh=tanh_tensor(zero_points=critical_points,scale=0.01).T #(N,m)
impulse=impulse_generator(a=critical_points).T #(N,m)
#tanh=1000*tanh/tanh.norm(dim=1,keepdim=True)
#tanh=tanh-torch.mean(tanh,dim=1,keepdim=True)
col_ones = torch.ones(10000,1)
#tanh = torch.cat([tanh, col_ones], dim=1)
p=svd(tensor=tanh)
#print(threshold_p@tanh.T)
#print(p@p.T)
print(off_diagonal_ratio(p@p.T))
#print(threshold_p@tanh)
plot(p)

#print(impulse.T@impulse)
print(off_diagonal_ratio(impulse.T@impulse))
#print(impulse.T@tanh)
#plot(impulse.T)



