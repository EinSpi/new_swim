import torch
import torch.nn.functional as F
from typing import Callable
from activations.activations import Activation
def total_loss(params:torch.Tensor, 
               x_points:torch.Tensor, y_targets:torch.Tensor, poles:torch.Tensor,
               activation:Activation):
    return loss(params, x_points, y_targets, poles, activation) + 0*regularization(params, x_points, poles, activation)+1e-6*l2_reg(params)

def loss(params:torch.Tensor, 
               x_points:torch.Tensor, y_targets:torch.Tensor, poles:torch.Tensor,
               activation:Activation):
    preds = activation.infer(x_points, params, poles)
    
    
    #cos based loss
    """
    dot_product = torch.sum(preds * y_targets)
    norm_preds = torch.norm(preds)
    norm_targets = torch.norm(y_targets)
    cosine_similarity = dot_product / (norm_preds * norm_targets + 1e-8)
    err=1.0 - cosine_similarity**2
    """
    
    
    #mse loss
    err=F.mse_loss(preds,y_targets)
    return err
    
    
def regularization(params:torch.Tensor, x_points:torch.Tensor, poles:torch.Tensor,
                   activation:Activation):
    x_dense = torch.linspace(x_points.min(), x_points.max(), 200)
    y_dense = activation.infer(x_dense, params,poles)
    return torch.mean(y_dense**2)
def l2_reg(params:torch.Tensor):
    return torch.norm(params, p=2) ** 2  # L2 范数平方
