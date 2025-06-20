import torch
from swim_backbones.base import BaseTorchBlock
from solvers import W_B_Solver,Probability_Solver,Adaptive_Solver
from activations import Activation
from utils.utils import clean_inputs, generate_x_point_pairs, find_nearest_indices

class Dense(BaseTorchBlock):
    def __init__(self, layer_width: int = 200, input_dimension: int=2, activation: Activation =None, device:torch.device=None,
                repetition_scaler: int = 2,
                gpu_gen: torch.Generator = None,
                cpu_gen: torch.Generator = None,

                w_b_solver: W_B_Solver = None,
                adaptive_solver: Adaptive_Solver = None,
                probability_solver: Probability_Solver = None,
                 ):
        super().__init__(layer_width, input_dimension, activation, device)

        self.repetition_scaler=repetition_scaler
        self.gpu_gen=gpu_gen
        self.cpu_gen=cpu_gen
        self.w_b_solver=w_b_solver
        self.adaptive_solver=adaptive_solver
        self.probability_solver=probability_solver
        self.register_buffer("a_params", torch.zeros(self.layer_width,self.activation.num_a_params))


    
    def fit(self, x:torch.Tensor,y:torch.Tensor):
        x=x.to(self.device)
        y=y.to(self.device)
        #make sure shape compatible, x(N,d) y(N,1)
        x,y =clean_inputs(x,y)
        x_pairs=generate_x_point_pairs(x=x,number=self.repetition_scaler*self.layer_width,generator=self.cpu_gen) #(m,2,d)
        #根据已定策略求：w，b，假性x插值点
        self.candidate_w_pool, self.candidate_b_pool,pseudo_samples=self.w_b_solver.compute_w_b_end_xy_pool(x_pairs)
        #根据假性x插值点，寻找最接近的真实x插值点以及对应的y插值点
        indices=find_nearest_indices(x_inter=pseudo_samples,x_real=x)#(N,k)
        self.candidate_x_pool,self.candidate_y_pool=x[indices],y[indices]#(N,k,d), (N,k,1)
        #如果是方差指导概率，先抽样，再算adpt，若不是，则需要算完adpt再算概率抽样
        if self.probability_solver.prob_strategy in ["var","uni","SWIM"]:
            #先抽样后拟合
            #求概率并抽样
            probability=self.probability_solver.probability_calculator(tensor1=x_pairs,tensor2=self.candidate_y_pool)
            sampled_indices= torch.multinomial(probability, num_samples=self.layer_width, replacement=False,generator=self.cpu_gen)
            
            #用抽到的索引选取池中项目
            self.weights,self.biases=self.candidate_w_pool[sampled_indices],self.candidate_b_pool[sampled_indices]
            #对抽到的对象进行a_param计算
            
            self.a_params=self.adaptive_solver.compute_a_paras(x_points=self.candidate_x_pool[sampled_indices],
                                                            w=self.weights,
                                                            b=self.biases,
                                                            y_points=self.candidate_y_pool[sampled_indices],
                                                            activation=self.activation,
                                                            )#(W,p+q)
            """
            #先拟合再抽样
            else:
                self.candidate_a_params_pool=self.adaptive_solver.compute_a_paras(x_points=self.candidate_x_pool,
                                                                w=self.candidate_w_pool,
                                                                b=self.candidate_b_pool,
                                                                y_points=self.candidate_y_pool,
                                                                activation=self.activation,
                                                                )
                probability=self.probability_solver.probability_calculator(tensor1=x_pairs,tensor2=self.candidate_y_pool)
                sampled_indices= torch.multinomial(probability, num_samples=self.layer_width, replacement=False,generator=self.gpu_gen)
                
                self.weights,self.biases=self.candidate_w_pool[sampled_indices],self.candidate_b_pool[sampled_indices]
                self.a_params=None if self.candidate_a_params_pool==None else self.candidate_a_params_pool[sampled_indices]
                
            """
                
        else:
            #整池计算a_params,然后抽样
            self.candidate_a_params_pool=self.adaptive_solver.compute_a_paras(x_points=self.candidate_x_pool,
                                                                              w=self.candidate_w_pool,
                                                                              b=self.candidate_b_pool,
                                                                              y_points=self.candidate_y_pool,
                                                                              activation=self.activation,
                                                                              )#(N,p+q)
            
            #计算整体优化好的函数在全域X上的表现，准备求概率
            F_matrix=self.activation.infer(x=self.candidate_w_pool@x.T+self.candidate_b_pool,a_params=self.candidate_a_params_pool).T
            #求概率并抽样
            probability=self.probability_solver.probability_calculator(tensor1=y,tensor2=F_matrix)
            sampled_indices= torch.multinomial(probability, num_samples=self.layer_width, replacement=False,generator=self.cpu_gen)
            #用抽到的索引选取池中项目
            self.weights,self.biases=self.candidate_w_pool[sampled_indices],self.candidate_b_pool[sampled_indices]
            self.a_params=None if self.candidate_a_params_pool==None else self.candidate_a_params_pool[sampled_indices]
    
        
        return self

    def forward(self, x:torch.Tensor):
        #x(T,d) self.weights (W,d) self.biases (W,1) self.a_params(W,p+q)
        x=x.to(self.device)
        linear_transformed_x=self.weights@x.T+self.biases #(W,T) +(W,1)= (W,T)
        return self.activation.infer(x=linear_transformed_x,a_params=self.a_params).T #(W,T),(W,p+q)->(W,T)->.T->(T,W)
        
        
    


    


        


    
    





    
    
    

    