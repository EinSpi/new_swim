from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import Callable

@dataclass
class Activation(ABC):
    num_a_params:int=1
    @abstractmethod
    def infer(self, x:torch.Tensor,a_params:torch.Tensor)->torch.Tensor:
        pass




@dataclass
class Rational(Activation):
    num_coeff_p:int=4
    num_coeff_q:int=3
    

    def __post_init__(self):
        # 自动设置父类的 num_a_params
        self.num_a_params = self.num_coeff_p + self.num_coeff_q

    def infer(self, x:torch.Tensor,a_params:torch.Tensor)->torch.Tensor:
        #预处理维度
        if a_params.dim() == 1:
            a_params = a_params.unsqueeze(0)     # (1, p+q)
        if x.dim()==1:
            x = x.unsqueeze(0) #(1, set_size)
        
        # x (B,set_size) a_params(B,p+q) B是批大小
        # 拆分分子/分母系数
        coeff_p = a_params[:,:self.num_coeff_p]         # shape (B,p)
        coeff_q = a_params[:,self.num_coeff_p:]         # shape (B,q)

        # 构造 分子：x 的幂次矩阵 与 coeff_p线性相乘
        x_powers_p = torch.stack([x ** i for i in range(self.num_coeff_p)], dim=2)  # shape (B, set_size, p)
        numerator = torch.bmm(x_powers_p,coeff_p.unsqueeze(2)).squeeze(2)#(B,set_size,p) bmm (B,p,1)= (B,set_size,1)->squeeze->(B,set_size)
        # 构造 分母：根据极点信息，制造不同的分母形式
        
        x_powers_q = torch.stack([x ** i for i in range(self.num_coeff_q)], dim=2)  # shape (B, set_size, q)
        denominator = 1+(torch.bmm(x_powers_q,coeff_q.unsqueeze(2)).squeeze(2))**2 # (B,set_size)
        
        res=numerator / denominator

        if res.shape[0]==1:
            res=res.squeeze(0) #(set_size,)
        
        return res  #(B,set_size)





@dataclass
class Adaptive_Activation(Activation):
    act:Callable=torch.relu
    def infer(self,x:torch.Tensor,a_params:torch.Tensor)->torch.Tensor:
        return self.act(x*a_params)
    
@dataclass
class Adpt_Tanh(Adaptive_Activation):
    def __post_init__(self):
        self.act=torch.tanh

@dataclass
class Adpt_Relu(Adaptive_Activation):
    def __post_init__(self):
        self.act=torch.relu

@dataclass
class Adpt_Sigmoid(Adaptive_Activation):
    def __post_init__(self):
        self.act=torch.sigmoid













@dataclass
class Non_Adaptive_Activation(Activation):
    act:Callable=torch.relu
    def __post_init__(self):
        self.num_a_params=0
    def infer(self,x:torch.Tensor,a_params:torch.Tensor):
        return self.act(x)
@dataclass
class Tanh(Non_Adaptive_Activation):
    def __post_init__(self):
        super().__post_init__()
        self.act=torch.tanh


@dataclass
class Relu(Non_Adaptive_Activation):
    def __post_init__(self):
        super().__post_init__()
        self.act=torch.relu

@dataclass
class Sigmoid(Non_Adaptive_Activation):
    def __post_init__(self):
        super().__post_init__()
        self.act=torch.sigmoid

   
    

    

