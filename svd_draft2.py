import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
import matplotlib.pyplot as plt

def svd(tensor:torch.Tensor):
    U, S, V = torch.linalg.svd(tensor, full_matrices=False)
    
    
    return U, S, V

def pseudo_inv_AAT(A:torch.Tensor):
    u,s,v=svd(A)
    s_inv=torch.diag(1.0 / s**2)
    return u@s_inv@u.T


A=torch.randn(10,4)
ps_inv_AAT=pseudo_inv_AAT(A)
projection=ps_inv_AAT@A
print(projection.T@projection)

