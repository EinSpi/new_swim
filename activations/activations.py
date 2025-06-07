from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import Callable

@dataclass
class Activation(ABC):
    num_a_params:int=1
    @abstractmethod
    def infer(self, x:torch.Tensor,a_params:torch.Tensor,poles:torch.Tensor=None):
        pass




@dataclass
class Rational(Activation):
    num_coeff_p:int=4
    num_coeff_q:int=3
    

    def __post_init__(self):
        # 自动设置父类的 num_a_params
        self.num_a_params = self.num_coeff_p + self.num_coeff_q

    def infer(self, x:torch.Tensor,a_params:torch.Tensor,poles:torch.Tensor=None):
        # x (N,) a_params(p,) p here should be equal to self.num_a_params。 poles是极点，要么None 要么 (2,)
        # 拆分分子/分母系数
        coeff_p = a_params[:self.num_coeff_p]         # shape (p,)
        coeff_q = a_params[self.num_coeff_p:]         # shape (q,)

        # 构造 分子：x 的幂次矩阵 与 coeff_p线性相乘
        x_powers_p = torch.stack([x ** i for i in range(len(coeff_p))], dim=1)  # shape (N, p)
        numerator = (x_powers_p @ coeff_p) # shape (N,)
        # 构造 分母：根据极点信息，制造不同的分母形式
        
        """
        if poles==None:
            #无极点，分母设计为一串偶次方和加一个正数，coeff_q的元素被用于(x-q_i)^n。具体是几次方观察num_coeffs_p, 
            #比如num_coeffs_p=3,说明分子二次多项式，那么我们在分母上采用二次方
            #比如num_coeffs_p=4,说明分子三次多项式，那么我们在分母上采用四次方
            even = self.num_coeff_p if self.num_coeff_p % 2 == 0 else self.num_coeff_p - 1 #计算所采用的次方数
            diff = x[:, None] - coeff_q[None, :]  # (N, q) 广播取（x-q_i）,作为偶数次幂的基底
            powered = diff.pow(even)              # (N, q) 上偶数次幂
            denominator = powered.sum(dim=1)+1           # (N,) 加和并加1， 构建分母形式：偶数次幂加和再加1
        else:
            #有指定极点，分母设计为固定极点形式，coeff_q用于挂载在分母前面充当权重即 (q1q2q3...qq)*(x-p1)^a*(x-p2)^a
            #其中极点的重数由分子确定
            #比如num_coeffs_p=3,说明分子二次多项式，那么a=1,分母2次
            #比如num_coeffs_p=4,说明分子三次多项式，那么a=2,分母4次
            even = self.num_coeff_p if self.num_coeff_p % 2 == 0 else self.num_coeff_p + 1
            pole_power=even//2 #求取极点重数
            denominator = torch.prod(coeff_q)*(x - poles[0]).pow(pole_power)*(x - poles[1]).pow(pole_power)# 固定极点，coeff_q全乘在一起挂在前面当系数
            
        
        """
        #PAU，分母绝对值
        x_powers_q = torch.stack([x ** i for i in range(len(coeff_q))], dim=1)  # shape (N, q)
        denominator = 1+(x_powers_q @ coeff_q)**2 # shape (N,)
        
        #分母跳幂保正
        #x_powers_q = torch.stack([x ** (2*i) for i in range(len(coeff_q))], dim=1)  # shape (N, q)
        #denominator = x_powers_q @ coeff_q
        
        return numerator / denominator  #(N,)





@dataclass
class Adaptive_Activation(Activation):
    act:Callable=torch.relu
    def infer(self,x:torch.Tensor,a_params:torch.Tensor,poles:torch.Tensor=None):
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
    def infer(self,x:torch.Tensor,a_params:torch.Tensor,poles:torch.Tensor=None):
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

   
    

    

