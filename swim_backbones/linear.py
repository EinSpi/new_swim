from .base import BaseTorchBlock
import torch
class Linear(BaseTorchBlock):
    def fit(self, x:torch.Tensor,y:torch.Tensor):
        #x (N,layer_width)
        #y (N,1)
        N=x.shape[0]
        ones_col=torch.ones((N, 1), dtype=x.dtype, device=x.device)
        x_aug=torch.cat([x, ones_col], dim=1) #(N,layer_width+1)
        w_b, *_ = torch.linalg.lstsq(x_aug, y) #()
        self.weights=w_b[:-1] #(layer_width,1)
        self.biases=w_b[-1] #(1,)
        return self
    
    def forward(self, x):
        #x(N，layer_width)
        #self.weights (layer_width,1) self.biases (1,)
        return x@self.weights+self.biases#(N,1)+(1,)=(N,1)s
        


