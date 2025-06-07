import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
import matplotlib.pyplot as plt

def tanh_tensor(zero_points:torch.Tensor,scale:float=50.0,length=1000):
    # 构造列坐标 0~999，形状为 (1, 1000)
    x = torch.arange(length).unsqueeze(0)  # shape: (1, 1000)

    # 广播生成偏移后的 tanh 输入，每行减去自己的零点位置
    # 结果 shape: (6, 1000)
    shifted_x = x - zero_points.unsqueeze(1)

    # 应用 tanh
    tanh_tensor = torch.tanh(shifted_x *scale)  # /50控制斜率平缓程度，可调
    return tanh_tensor
def condition_number(x:torch.Tensor):
    #x(N,m)
    U,S,V=torch.linalg.svd(x)
    return torch.max(S)/torch.min(S)

def critical_points(l,r,step):
    return torch.arange(l, r + 1e-8, step)

zero_points=torch.tensor([100,200,300,400,500,600,700])
basis= tanh_tensor(zero_points=zero_points).T#(1000,7)

y=tanh_tensor(zero_points=torch.tensor([205,502,598])).T#(1000,3)
y=torch.sum(y,dim=1,keepdim=True)#(1000,1)


Q,R=torch.linalg.qr(basis)
_,_,V=torch.linalg.svd(torch.cat([basis,y],dim=1),full_matrices=True)
#print(V.shape)
v=V[-1]
#print(v.shape)
v=v/(-v[-1])
w_svd=v[:-1]
w_lstsq,*_=torch.linalg.lstsq(basis,y)

w_qr= torch.linalg.solve(R, Q.T@y)
#print(v)
print(w_svd)
print(w_qr)
print(w_lstsq)



