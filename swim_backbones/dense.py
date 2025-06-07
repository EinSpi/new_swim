from dataclasses import dataclass,field
from typing import Callable, Union
import numpy as np
from .base import BaseTorchBlock
from utils.utils import generate_point_sets, w_b_calculator, clean_inputs, general_forward, column_normalize_and_append_1,probability_calculator,delete_tensors, generate_x_point_pairs,batch_extend_lines_to_box,batch_interpolate_between, find_nearest_indices,a_params_initializer
from utils.a_para_optimizer_loss_utils import total_loss
import torch
from torchmin import minimize
from w_b_solver.w_b_solver import W_B_Solver
from pole_supression.pole_supression import Pole_Supression
import matplotlib.pyplot as plt

@dataclass
class Dense(BaseTorchBlock):
    random_seed: int = 1
    repetition_scaler: int = 2
    set_size:int =6
    lower:torch.Tensor=torch.tensor([0,-20],dtype=torch.float32)
    upper:torch.Tensor=torch.tensor([40,20],dtype=torch.float32)
    w_b_solver: W_B_Solver = field(init=False)
    poles_supressor: Pole_Supression = field(default_factory=lambda: Pole_Supression(pole_policy="no_poles"))
    def __post_init__(self):
        super().__post_init__()
        self.w_b_solver = W_B_Solver(
            lower=self.lower,
            upper=self.upper,
            num_interpolation_per_pair=self.set_size,  
            reference="sample"
        )
    

    
    def fit(self, x:torch.Tensor,y:torch.Tensor):
        
        #prepare random generator
        generator = torch.Generator()
        #generator.manual_seed(self.random_seed)

        #make sure shape compatible, x(N,d) y(N,1)
        x,y =clean_inputs(x,y)
        #random select m k-sized point-sets and its corresponding y value.


        """
        self.candidate_x_pool,self.candidate_y_pool=generate_point_sets(x=x,y=y,
                                                    size=self.set_size,
                                                    number=self.repetition_scaler*self.layer_width,
                                                    generator=generator)#(m,k,d) #(m,k,1)
                                                    
        
        #calculate w pool and b pool
        self.candidate_w_pool, self.candidate_b_pool =w_b_calculator(self.candidate_x_pool) #(m,d) #(m,1)
        """
        

        print(self.repetition_scaler*self.layer_width)
        x_pairs=generate_x_point_pairs(x=x,number=self.repetition_scaler*self.layer_width,generator=generator,random_seed=42) #(m,2,d)
        print(x_pairs.shape)
        #根据已定策略求：w，b，端点，xy抽样池
        self.candidate_w_pool, self.candidate_b_pool,self.candidate_ends_pool,pseudo_samples=self.w_b_solver.compute_w_b_end_xy_pool(x_pairs)
        #print(self.candidate_w_pool.shape)
        #print(self.candidate_b_pool.shape)
        indices=find_nearest_indices(x_inter=pseudo_samples,x_real=x)#(m,k)
        self.candidate_x_pool,self.candidate_y_pool=x[indices],y[indices]
        #根据 w,b,端点求激活函数的极点位置，把它推到端点之外，以防激活函数出现极点，如果没有极点，则按照没极点的激活函数parametrization进行
        self.candidate_poles_pool = self.poles_supressor.compute_poles(self.candidate_w_pool, self.candidate_b_pool,self.candidate_ends_pool) #(m,2)
       
        #self.candidate_poles_pool.requires_grad=False#强制极点位置不参与优化
        #根据 xy抽样池,w,b,以及极点位置对所有激活函数的参数进行优化
        self.a_params_calculator()
        has_bad = torch.isnan(self.candidate_a_params_pool).any() or torch.isinf(self.candidate_a_params_pool).any()
        if has_bad:
            print("bad trained a_params")
        delete_tensors(self.candidate_x_pool,self.candidate_y_pool)

        #计算整体优化好的函数在全域X上的表现，准备SVD
        inte_pool_forward=general_forward(x=x,w=self.candidate_w_pool,b=self.candidate_b_pool,
                                          a_params=self.candidate_a_params_pool, poles=self.candidate_poles_pool,activation=self.activation)#(N,m)
        #控制所有模长固定，加一列全1
        inte_pool_forward=column_normalize_and_append_1(inte_pool_forward)#(N,m+1)
        print(torch.max(inte_pool_forward))
        print(torch.min(inte_pool_forward))

        #不控制模长，直接append全1
        #ones_col = torch.ones((inte_pool_forward.shape[0], 1), dtype=inte_pool_forward.dtype, device=inte_pool_forward.device)
        #拼接
        #inte_pool_forward = torch.cat([inte_pool_forward, ones_col], dim=1)  # shape: (N, m+1)
        
        #用刚刚的全域X上的表现计算概率
        #probability=probability_calculator(inte_pool_forward=inte_pool_forward,y=y,full_matrices=False)#(m,)
        #用点集y方差定义probability y (m,k,1)
        logits=torch.var(self.candidate_y_pool, dim=1, unbiased=False).squeeze(-1)
        probability=logits/torch.sum(logits)

        #用概率指导抽样
        sampled_indices= torch.multinomial(probability, num_samples=self.layer_width, replacement=False) #(self.layer_width,)
        #根据抽样抽出来的索引从pool中选择选中的东西
        self.weights,self.biases=self.candidate_w_pool[sampled_indices],self.candidate_b_pool[sampled_indices]
        self.a_params=None if self.candidate_a_params_pool==None else self.candidate_a_params_pool[sampled_indices]
        self.poles= None if self.candidate_poles_pool==None else self.candidate_poles_pool[sampled_indices]
        delete_tensors(self.candidate_w_pool,self.candidate_b_pool,self.candidate_a_params_pool,self.candidate_ends_pool,self.candidate_poles_pool)
        # self.weights (self.layer_width,d) self.biases (self.layer_width,1) self.a_params(self.layer_width,p)

        return self



    
    def forward(self, x):
        #x(N,d) self.weights (self.layer_width,d) self.biases (self.layer_width,1) self.a_params(self.layer_width,p)
        return general_forward(x=x,w=self.weights,b=self.biases,a_params=self.a_params,poles=self.poles, activation=self.activation)#(N,self.layer_width)
        
    


    def a_params_calculator(self):
        m=self.candidate_x_pool.shape[0]#candidate set number
        print("candidate_x_pool_shape")
        print(self.candidate_x_pool.shape)
        candidate_x_pool_1d=torch.matmul(self.candidate_x_pool,self.candidate_w_pool.unsqueeze(-1))+self.candidate_b_pool.unsqueeze(1) # (m,k,1) + (m,1,1)=(m,k,1)
        print("point_set_shape")
        print(candidate_x_pool_1d.shape)
        num_a_params=self.activation.num_a_params
        if num_a_params==0:
            a_params=None
        else:
            final_a_params= torch.zeros(m, num_a_params, dtype=torch.float32)#空容器用来装优化出来的parameter
            for i in range(m):
                x_points = candidate_x_pool_1d[i].squeeze(-1) #(k,1)->(k,)
                y_targets = self.candidate_y_pool[i].squeeze(-1) #(k,1) ->(k,)

                poles = None if self.candidate_poles_pool==None else self.candidate_poles_pool[i] #(2,)

                #init_params = torch.zeros(self.activation.num_a_params, dtype=torch.float32, requires_grad=True)
                #init_params = torch.cat([torch.zeros(self.activation.num_a_params-3,dtype=torch.float32),torch.ones(3,dtype=torch.float32)]).requires_grad_()
                a_params = torch.tensor([0.0218, 0.5, 1.5957, 1.1915, 0.0, 0.0, 2.383], dtype=torch.float32,requires_grad=True)
                #init_params = a_params_initializer(x_points=x_points,y_targets=y_targets,num_coeff_p=3,num_coeff_q=3)
                #has_bad = torch.isnan(init_params).any() or torch.isinf(init_params).any()
                #if has_bad:
                    #print("bad initial a params")
                #init_params = torch.tensor([1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0], dtype=torch.float32,requires_grad=True)
                def loss_fn(params):
                    return total_loss(params, x_points, y_targets, poles, self.activation)
                """
                #prepare Adams and optimize
                optimizer = torch.optim.Adam([a_params], lr=1e-2)
                best_loss,no_improve_steps = float('inf'),0
                for step in range(700):
                    optimizer.zero_grad()
                    loss = loss_fn(a_params)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss - 1e-6:
                        best_loss = loss.item()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1

                    if no_improve_steps > 200:
                        print(f"Early stopped at step {step} (no improvement for {200} steps)")
                        break
                result=a_params.detach()
                """
                #L-BFGS method
                result = minimize(loss_fn, a_params, method='l-bfgs', tol=1e-9, max_iter=10000)
                result = result.x.detach()
                
                if torch.isnan(result).any():
                    print("bad trained a_params (nan)")
                if torch.isinf(result).any():
                    print("bad trained a_params (inf)")
                if i%(m//8)==0:
                    #定期画一下我的sub optimization情况图
                    #look at a slice of sub optimization, plot
                    x_range=torch.linspace(x_points.min(), x_points.max(), 200) #(200,)
                    y_range=self.activation.infer(x=x_range,a_params=result)
                    plt.figure(figsize=(8, 5))
                    plt.scatter(x_points, y_targets, label='Data (x vs y)', color='blue')
                    plt.plot(x_range, y_range, label='rational(x, a)', color='red')
                    plt.xlabel('x')
                    plt.ylabel('y / rational(x, a)')
                    plt.legend()
                    plt.title('Data vs Function Curve')
                    plt.grid(True)
                    save_path = "Results/"+"rational"+str(i//(m//8))
                    plt.savefig(save_path)
                    plt.close()  # 很重要！防止内存累积
                #print(result.shape)
                final_a_params[i] = result
        self.candidate_a_params_pool=final_a_params#(m,p)


        


    
    





    
    
    

    