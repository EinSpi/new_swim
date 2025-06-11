import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
import matplotlib.pyplot as plt

def tanh_tensor(zero_points:torch.Tensor,scale:float=50.0):
    # 构造列坐标 0~999，形状为 (1, 1000)
    x = torch.arange(10000).unsqueeze(0)  # shape: (1, 1000)

    # 广播生成偏移后的 tanh 输入，每行减去自己的零点位置
    # 结果 shape: (6, 1000)
    shifted_x = x - zero_points.unsqueeze(1)

    # 应用 tanh
    tanh_tensor = torch.tanh(shifted_x / scale)  # /50控制斜率平缓程度，可调
    return tanh_tensor

def plot(tensor:torch.Tensor):
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

tanh=tanh_tensor(zero_points=torch.tensor([ 3500,3501,6000 ], dtype=torch.float32),scale=5.0)#(m,N)
print(tanh.shape)

Q, _ = torch.linalg.qr(tanh.T, mode='reduced')  #(N,m)

print(Q.T@tanh.T)
plot(Q.T)

