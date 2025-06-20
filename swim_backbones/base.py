from abc import ABC, abstractmethod
import torch.nn as nn 
from activations import Activation
import torch


class BaseTorchBlock(nn.Module,ABC):
    def __init__(self, layer_width: int = 200, input_dimension: int=2, activation: Activation =None, device:torch.device=None):
        super().__init__()
        self.layer_width=layer_width
        self.input_dimension=input_dimension
        self.activation=activation
        self.device=device
        self.register_buffer("weights", torch.zeros(self.layer_width,self.input_dimension))
        self.register_buffer("biases", torch.zeros(self.layer_width,1))

    ################################################
    ######abstract placeholder to for training######
    ################################################
    @abstractmethod
    def fit(self,x,y=None):
        pass
    @abstractmethod
    def forward(self,x):
        pass
    
class TorchPipeline(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def fit(self, X, y=None):
        for module in self.modules_list:
            module.fit(X, y)   
            X = module(X)  # forward一次得到下一级输入
        return self

    def forward(self, X):
        for module in self.modules_list:
            X = module(X)
        return X
    
class Swim_Model(TorchPipeline):
    def __init__(self, modules):
        super().__init__(modules=modules)

    
    

    
    
    