from dataclasses import dataclass
from activations.activations import Activation
import torch
import torch.nn.functional as F
from torchmin import minimize
import matplotlib.pyplot as plt

    
    
def mse_loss(params:torch.Tensor, x_points:torch.Tensor, y_targets:torch.Tensor, activation:Activation):
    preds = activation.infer(x_points, params) 
    err=F.mse_loss(preds,y_targets,reduction="mean")
    return err

def cos_loss(params:torch.Tensor, x_points:torch.Tensor, y_targets:torch.Tensor, activation:Activation):
    #cos based loss
    preds = activation.infer(x_points, params)#(B,set_size)
    dot_product = torch.sum(preds * y_targets,dim=-1) #(B,)或标量
    norm_preds = torch.norm(preds,dim=-1) #(B,)或标量
    norm_targets = torch.norm(y_targets, dim=-1) #(B,)或标量
    cosine_similarity = dot_product / (norm_preds * norm_targets + 1e-8)#(B,)或标量
    err=1.0 - cosine_similarity**2 #(B,)或标量
    err=torch.mean(err)
    return err

def regularization(params:torch.Tensor, x_points:torch.Tensor, activation:Activation):
    
    if params.dim()==1:
        x_dense = torch.linspace(x_points.min(), x_points.max(), 200)
        y_dense = activation.infer(x_dense, params)
    else:
        B = params.shape[0]
        # 构造共享 x_dense：shape (B, 200)
        x_min = x_points.min(dim=-1).values    # (B,)
        x_max = x_points.max(dim=-1).values    # (B,)
        x_dense = torch.stack([torch.linspace(x_min[i], x_max[i], 200) for i in range(B)])  # shape: (B, 200)
        y_dense = activation.infer(x_dense, params) #(B,200)

    return torch.mean(y_dense**2)

def l2_reg(params:torch.Tensor):
    return (params ** 2).sum(dim=-1).mean()  # L2 范数平方

def random_visualize(x_1d:torch.Tensor, y_points:torch.Tensor,a_params:torch.Tensor, activation:Activation,generator:torch.Generator, device:torch.device,how_many:int=8, save_path:str=" "):
    #create random slices to plot
    #x_1d (B,set_size,1)
    
    view_indices = torch.randperm(x_1d.shape[0],generator=generator)[:how_many]
    x_slices=x_1d[view_indices].squeeze(-1) #(how_many, set_size)
    y_slices=y_points[view_indices].squeeze(-1) #(how_many, set_size)
    a_slices=a_params[view_indices]#(how_many, p+q)
    for i in range(how_many):
        x_range=torch.linspace(x_slices[i].min(), x_slices[i].max(), 200,device=device) #(200,)
        y_range=activation.infer(x=x_range,a_params=a_slices[i])#(200,)
        # 将用于绘图的 Tensor 转为 CPU
        x_cpu = x_slices[i].cpu().numpy()
        y_cpu = y_slices[i].cpu().numpy()
        x_range_cpu = x_range.cpu().numpy()
        y_range_cpu = y_range.cpu().numpy()
        #绘图
        plt.figure(figsize=(8, 5))
        plt.scatter(x_cpu, y_cpu, label='Data (x vs y)', color='blue')
        plt.plot(x_range_cpu, y_range_cpu, label='rational(x, a)', color='red')
        plt.xlabel('x')
        plt.ylabel('y / rational(x, a)')
        plt.legend()
        plt.title('Data vs Function Curve')
        plt.grid(True)
        plt.savefig(save_path+"/"+"rational"+str(i))
        plt.close()


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
    loss_metric:str="mse"
    reg_factor:float=1e-6
    int_sketch:bool=True
    optimizer:str="lbfgs"
    lr:float=1e-3
    max_epochs:int=3000
    cpu_gen:torch.Generator=torch.Generator()
    save_path:str=" "
    device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def compute_a_paras(self,x_points:torch.Tensor,w:torch.Tensor,b:torch.Tensor,y_points:torch.Tensor,activation:Activation):
        x_1d=torch.matmul(x_points,w.unsqueeze(-1))+b.unsqueeze(1) # (N,k,1) + (N,1,1)=(N,k,1)
        #if cuda available, use gpu
        #optimize
        if activation.num_a_params==0:
            #无a_params
            result=None
        elif self.optimizer=="lbfgs":
            #使用lbfgs优化器
            result=self.lbfgs_optimize(x_1d=x_1d,y_points=y_points,activation=activation)
        elif self.optimizer=="adam":
            #使用adams优化器求解
            result=self.adam_optimize(x_1d=x_1d,y_points=y_points,activation=activation)
        else:
            raise ValueError("invalid optimizer")
        
        if self.int_sketch:
            #随机可视化拟合情况
            random_visualize(x_1d=x_1d,y_points=y_points,a_params=result,activation=activation,generator=self.cpu_gen,how_many=8,save_path=self.save_path,device=self.device)
            
        return result
    
    def lbfgs_optimize(self, x_1d:torch.Tensor,y_points:torch.Tensor,activation:Activation)->torch.Tensor:
        #x_1d (B,set_size,1) y_points (B,set_size, 1)
        m=x_1d.shape[0]
        print(m)
        final_a_params= torch.zeros(m, activation.num_a_params, dtype=torch.float32,device=self.device)#空容器用来装优化出来的parameter
        for i in range(m):
            print(i)
            x_slice,y_slice=x_1d[i].squeeze(-1),y_points[i].squeeze(-1)
            init_params = torch.tensor([0.0218, 0.5, 1.5957, 1.1915, 0.0, 0.0, 2.383], dtype=torch.float32,requires_grad=True,device=self.device)
            
            def loss_fn(params):
                if self.loss_metric=="mse":
                    return mse_loss(params, x_slice, y_slice, activation) +self.reg_factor*l2_reg(params)
                elif self.loss_metric=="cos":
                    return cos_loss(params, x_slice, y_slice, activation) +self.reg_factor*l2_reg(params)
                else:
                    raise ValueError("undefined loss metrics")
            #L-BFGS method
            result = minimize(loss_fn, init_params, method='l-bfgs', tol=1e-9, max_iter=self.max_epochs)
            result = result.x.detach()
            
            if torch.isnan(result).any():
                print("bad trained a_params (nan)")
            if torch.isinf(result).any():
                print("bad trained a_params (inf)")
            
            final_a_params[i] = result
        return final_a_params
    
    def adam_optimize(self, x_1d:torch.Tensor, y_points:torch.Tensor,activation:Activation)->torch.Tensor:
        m=x_1d.shape[0]
        print(m)
        a_params = torch.tensor([0.0218, 0.5, 1.5957, 1.1915, 0.0, 0.0, 2.383],device=self.device).repeat(m, 1).clone().detach().requires_grad_(True)
        optimizer=torch.optim.Adam([a_params],lr=self.lr)
        best_loss = float('inf')
        best_params = None
        best_epoch =0
        patience=self.max_epochs//5
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()

            # loss 
            if self.loss_metric=="mse":
                loss = mse_loss(a_params, x_1d.squeeze(-1), y_points.squeeze(-1), activation)  + self.reg_factor * l2_reg(a_params)
            elif self.loss_metric=="cos":
                loss = cos_loss(a_params, x_1d.squeeze(-1), y_points.squeeze(-1), activation) + self.reg_factor* l2_reg(a_params)
            else:
                raise ValueError("undefined loss metrics")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = a_params.detach().clone()
                best_epoch=epoch
            elif epoch-best_epoch>=patience:
                #early stop
                print(f"Early stopping at epoch {epoch}, best was at {best_epoch} with loss {best_loss:.6f}")
                break
            loss.backward()
            optimizer.step()

        return best_params




        
    

       