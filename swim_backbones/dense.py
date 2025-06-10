from dataclasses import dataclass,field
from .base import BaseTorchBlock
from utils.utils import clean_inputs, general_forward,delete_tensors, generate_x_point_pairs, find_nearest_indices
import torch
from w_b_solver.w_b_solver import W_B_Solver
from probability_solver.probability_solver import Probability_Solver
from adaptive_solver.adaptive_solver import Adaptive_Solver

@dataclass
class Dense(BaseTorchBlock):
    random_seed: int = 1
    repetition_scaler: int = 2
    set_size:int =6
    loss_metric:str="mse"
    reg_factor_1:float=0.0
    reg_factor_2:float=1e-6
    prob_strategy:str = "var"

    w_b_solver: W_B_Solver = field(init=False)
    adaptive_solver:Adaptive_Solver = field(init=False)
    probability_solver:Probability_Solver = field(init=False)
    
    def __post_init__(self):
        super().__post_init__()
        self.w_b_solver = W_B_Solver(num_interpolation_per_pair=self.set_size)
        self.adaptive_solver=Adaptive_Solver(loss_metric=self.loss_metric,reg_factor_1=self.reg_factor_1,reg_factor_2=self.reg_factor_2)
        self.probability_solver=Probability_Solver(prob_strategy=self.prob_strategy)


    
    def fit(self, x:torch.Tensor,y:torch.Tensor):
        
        #prepare random generator
        generator = torch.Generator()
        #generator.manual_seed(self.random_seed)

        #make sure shape compatible, x(N,d) y(N,1)
        x,y =clean_inputs(x,y)
        x_pairs=generate_x_point_pairs(x=x,number=self.repetition_scaler*self.layer_width,generator=generator,random_seed=42) #(m,2,d)
        #根据已定策略求：w，b，假性x插值点
        self.candidate_w_pool, self.candidate_b_pool,pseudo_samples=self.w_b_solver.compute_w_b_end_xy_pool(x_pairs)
        #根据假性x插值点，寻找最接近的真实x插值点以及对应的y插值点
        indices=find_nearest_indices(x_inter=pseudo_samples,x_real=x)#(N,k)
        self.candidate_x_pool,self.candidate_y_pool=x[indices],y[indices]#(N,k,d)
        #如果是方差指导概率，先抽样，再算adpt，若不是，则需要算完adpt再算概率抽样
        if self.prob_strategy=="var":
            #求概率并抽样
            probability=self.probability_solver.probability_calculator(tensor1=x_pairs,tensor2=self.candidate_y_pool)
            sampled_indices= torch.multinomial(probability, num_samples=self.layer_width, replacement=False)
            #用抽到的索引选取池中项目
            self.weights,self.biases=self.candidate_w_pool[sampled_indices],self.candidate_b_pool[sampled_indices]
            #对抽到的对象进行a_param计算
            self.a_params=self.adaptive_solver.compute_a_paras(x_points=self.candidate_x_pool[sampled_indices],
                                                               w=self.weights,
                                                               b=self.biases,
                                                               y_points=self.candidate_y_pool[sampled_indices],
                                                               activation=self.activation)#(W,p+q)
        else:
            #整池计算a_params,然后抽样
            self.candidate_a_params_pool=self.adaptive_solver.compute_a_paras(x_points=self.candidate_x_pool,
                                                                              w=self.candidate_w_pool,
                                                                              b=self.candidate_b_pool,
                                                                              y_points=self.candidate_y_pool,
                                                                              activation=self.activation)#(N,p+q)
            
            #计算整体优化好的函数在全域X上的表现，准备求概率
            F_matrix=general_forward(x=x,w=self.candidate_w_pool,b=self.candidate_b_pool,
                                              a_params=self.candidate_a_params_pool,activation=self.activation)#(T,N)
            #求概率并抽样
            probability=self.probability_solver.probability_calculator(tensor1=y,tensor2=F_matrix)
            sampled_indices= torch.multinomial(probability, num_samples=self.layer_width, replacement=False)
            #用抽到的索引选取池中项目
            self.weights,self.biases=self.candidate_w_pool[sampled_indices],self.candidate_b_pool[sampled_indices]
            self.a_params=None if self.candidate_a_params_pool==None else self.candidate_a_params_pool[sampled_indices]
            delete_tensors(self.candidate_a_params_pool)

        delete_tensors(self.candidate_w_pool,self.candidate_b_pool,self.candidate_x_pool,self.candidate_y_pool)
        return self

    def forward(self, x):
        #x(N,d) self.weights (self.layer_width,d) self.biases (self.layer_width,1) self.a_params(self.layer_width,p)
        return general_forward(x=x,w=self.weights,b=self.biases,a_params=self.a_params, activation=self.activation)#(N,self.layer_width)
        
    


    


        


    
    





    
    
    

    