from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch.nn as nn 
from activations.activations import Activation

@dataclass
class BaseTorchBlock(nn.Module,ABC):
    layer_width: int = None
    activation: Activation =None
    weights: np.ndarray = None
    biases: np.ndarray = None
    def __post_init__(self):
        super().__init__()

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

    
    

    
    
    