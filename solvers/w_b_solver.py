from dataclasses import dataclass
import torch

@dataclass
class W_B_Solver:
    s1:float=-1.0
    s2:float=1.0
    num_interpolation_per_pair:int=20

    def compute_w_b_end_xy_pool(self, point_pairs:torch.Tensor):
       
        #生成x点抽样池，在参考点之间随机抽取n个点
        m,_,_=point_pairs.shape
        start=point_pairs[:,0,:]#(m,d)
        end=point_pairs[:,1,:]#(m,d)
        # 随机生成权重 t ∈ [0,1]，shape: (m, k, 1)
        #t = torch.rand(m, self.num_interpolation_per_pair, 1, device=point_pairs.device, dtype=point_pairs.dtype)
        # 均匀生成t: (m, k, 1)，均匀分布于 [0,1]
        t = torch.linspace(0, 1, steps=self.num_interpolation_per_pair, device=point_pairs.device, dtype=point_pairs.dtype)
        t = t.view(1, -1, 1).expand(m, -1, -1)#(m, k, 1)
        # 插值：start.unsqueeze(1) shape → (m,1,d)
        pseudo_samples = (1 - t) * start.unsqueeze(1) + t * end.unsqueeze(1)  # (m, k, d)

        
        #solve w and b so that wx1+b=anchor_1, wx2+b=anchor_2
        w = start-end #(m,d)求差
        #print(w.shape)
        norm_sq = (w ** 2).sum(dim=1, keepdim=True)  # (m, 1)
        w = (self.s2-self.s1)* (w / norm_sq)  #(m,d)
        b = -(w*end).sum(dim=1,keepdim=True)+self.s1 #-wTx1+s1 (m,1)+(1,)=(m,1)
        
        return w,b,pseudo_samples #(m,1)*(m,d)->(m,d), (m,1) (m,2,d) (m,n,d)

       
            


