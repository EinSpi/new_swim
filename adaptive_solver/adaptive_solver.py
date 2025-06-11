from dataclasses import dataclass
from activations.activations import Activation
import torch
import torch.nn.functional as F
from torchmin import minimize
import matplotlib.pyplot as plt

    
    
def mse_loss(params:torch.Tensor, x_points:torch.Tensor, y_targets:torch.Tensor, activation:Activation):
    preds = activation.infer(x_points, params)
    err=F.mse_loss(preds,y_targets)
    return err

def cos_loss(params:torch.Tensor, x_points:torch.Tensor, y_targets:torch.Tensor, activation:Activation):
    #cos based loss
    preds = activation.infer(x_points, params)
    dot_product = torch.sum(preds * y_targets)
    norm_preds = torch.norm(preds)
    norm_targets = torch.norm(y_targets)
    cosine_similarity = dot_product / (norm_preds * norm_targets + 1e-8)
    err=1.0 - cosine_similarity**2
    return err

def regularization(params:torch.Tensor, x_points:torch.Tensor, activation:Activation):
    x_dense = torch.linspace(x_points.min(), x_points.max(), 200)
    y_dense = activation.infer(x_dense, params)
    return torch.mean(y_dense**2)

def l2_reg(params:torch.Tensor):
    return torch.norm(params, p=2) ** 2  # L2 范数平方

@dataclass
class Adaptive_Solver:
    """
        返回adaptive_parameters,可以是抽样完的，也可以是还没抽的总池。

        Args:
            x_points (torch.Tensor): x插值点 (N,k,d)
            y_points (torch.Tensor): y插值点 (N,k,1)
            w: 权重 (N,d)
            b: 偏移 (N,1)

        Returns:
            prob (torch.Tensor): 概率分布 (N,)
    """
    loss_metric:str="cos"
    reg_factor_1:float=0.0
    reg_factor_2:float=1e-6
    int_sketch:bool=True
    def compute_a_paras(self,x_points:torch.Tensor,w:torch.Tensor,b:torch.Tensor,y_points:torch.Tensor,activation:Activation,save_path:str):
        m=x_points.shape[0]
        x_1d=torch.matmul(x_points,w.unsqueeze(-1))+b.unsqueeze(1) # (N,k,1) + (N,1,1)=(N,k,1)
        num_a_params=activation.num_a_params
        if num_a_params==0:
            return None
        else:
            final_a_params= torch.zeros(m, num_a_params, dtype=torch.float32)#空容器用来装优化出来的parameter
            for i in range(m):
                x_slice,y_slice=x_1d[i].squeeze(-1),y_points[i].squeeze(-1)
                init_params = torch.tensor([0.0218, 0.5, 1.5957, 1.1915, 0.0, 0.0, 2.383], dtype=torch.float32,requires_grad=True)
               
                def loss_fn(params):
                    if self.loss_metric=="mse":
                        return mse_loss(params, x_slice, y_slice, activation) + self.reg_factor_1*regularization(params, x_slice, activation)+self.reg_factor_2*l2_reg(params)
                    elif self.loss_metric=="cos":
                        return cos_loss(params, x_slice, y_slice, activation) + self.reg_factor_1*regularization(params, x_slice, activation)+self.reg_factor_2*l2_reg(params)
                    else:
                        raise ValueError("undefined loss metrics")
                #L-BFGS method
                result = minimize(loss_fn, init_params, method='l-bfgs', tol=1e-9, max_iter=10000)
                result = result.x.detach()
                
                if torch.isnan(result).any():
                    print("bad trained a_params (nan)")
                if torch.isinf(result).any():
                    print("bad trained a_params (inf)")
                #绘制中间激活函数图像
                if self.int_sketch and i%(m//8)==0:
                    #定期画一下我的sub optimization情况图
                    #look at a slice of sub optimization, plot
                    x_range=torch.linspace(x_slice.min(), x_slice.max(), 200) #(200,)
                    y_range=activation.infer(x=x_range,a_params=result)
                    plt.figure(figsize=(8, 5))
                    plt.scatter(x_slice, y_slice, label='Data (x vs y)', color='blue')
                    plt.plot(x_range, y_range, label='rational(x, a)', color='red')
                    plt.xlabel('x')
                    plt.ylabel('y / rational(x, a)')
                    plt.legend()
                    plt.title('Data vs Function Curve')
                    plt.grid(True)
                    plt.savefig(save_path+"/"+"rational"+str(i//(m//8)))
                    plt.close()  # 很重要！防止内存累积
                #print(result.shape)
                final_a_params[i] = result
            return final_a_params
        
    

       