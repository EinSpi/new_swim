from swim_backbones.base import BaseTorchBlock
import torch

class Linear(BaseTorchBlock):

    def fit(self, x:torch.Tensor,y:torch.Tensor):
        #x (T,N)
        #y (T,1)
        x=x.to(self.device)
        y=y.to(self.device)
        T=x.shape[0]
        ones_col=torch.ones((T, 1), dtype=x.dtype, device=self.device)
        x_aug=torch.cat([x, ones_col], dim=1) #(T,N+1)
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        view_indices = torch.randperm(x_aug.shape[1],generator=generator)[:5]
        print("x_aug:")
        print(x_aug[:,view_indices])
        print("x_aug data type:")
        print(x_aug.dtype)          # 应为float32/float64
        print("x_aug device:")
        print(x_aug.device)        # 确保同一设备
        print("x_aug grad:")
        print(x_aug.requires_grad)  # 是否在计算图中
        print("y:")
        print(y)
        print("y data type:")
        print(y.dtype)
        print("y device:")
        print(y.device)
        print("y grad:")
        print(y.requires_grad)
        """
        """
        _,_,Vh=torch.linalg.svd(torch.cat([x_aug, y], dim=1),full_matrices=True)#(T,N+2)
        v=Vh[-1]#(N+2)
        v = v / (-v[-1])#把最后一个调为-1
        self.weights=v[:-2].unsqueeze(-1)#(N,1)
        print(self.weights.shape)
        self.biases=v[-2].unsqueeze(-1)#(1,1)
        print(self.biases.shape)
        """
        #gelsd
        w_b, *_ = torch.linalg.lstsq(x_aug,y,driver="gels") 
        self.weights=w_b[:-1].T #(layer_width,input_dimension)
        self.biases=w_b[-1:].T #(layer_width,1)
        
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        view_indices = torch.randperm(self.weights.shape[0],generator=generator)[:5]
        print("weights:")
        print(self.weights[view_indices])
        """

        return self
    
    def forward(self, x):
        #x(B，input_dimension)
        #self.weights (layer_width,input_dimension) self.biases (layer_width,1)
        x.to(self.device)
        return x@self.weights.T+self.biases#(B,layer_width)+(layer_width,1)=(B,1)
        


