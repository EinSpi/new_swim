from dataclasses import dataclass
import torch

def batch_extend_lines_to_box(pairs: torch.Tensor, 
                              lower: torch.Tensor, 
                              upper: torch.Tensor) -> torch.Tensor:
    """
    批量计算 m 对点组成的直线与 box 相交的两个端点。
    
    参数:
        pairs: (m, 2, d) - 每对 (a, b)
        lower: (d,) - box 的下界
        upper: (d,) - box 的上界

    返回:
        endpoints: (m, 2, d) - 每行是两个点 (p_min, p_max)
    """
    a = pairs[:, 0, :]  # (m, d)
    b = pairs[:, 1, :]  # (m, d)
    d_vec = b - a       # (m, d)

    # 避免除零：设置为非常大的正负数用于广播裁剪（后续用 where 过滤）
    eps = 1e-12
    d_safe = d_vec.clone()
    d_safe[d_safe.abs() < eps] = eps

    t1 = (lower - a) / d_safe  # (m, d)
    t2 = (upper - a) / d_safe  # (m, d)
    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)

    t_low = t_min.max(dim=1).values.unsqueeze(1)  # (m, 1)
    t_high = t_max.min(dim=1).values.unsqueeze(1) # (m, 1)

    p1 = a + t_low * d_vec  # (m, d)
    p2 = a + t_high * d_vec # (m, d)

    return torch.stack([p1, p2], dim=1)  # (m, 2, d)


@dataclass
class W_B_Solver:
    w_adaptive:bool=True
    w_scale:float=1.0
    reference:str="sample"#whether we use the sampled data point pair A,B OR end points E,F as reference to compute w and b
    upper:torch.Tensor=None
    lower:torch.Tensor=None
    s1:float=-1.0
    s2:float=1.0
    num_interpolation_per_pair:int=20

    def compute_w_b_end_xy_pool(self, point_pairs:torch.Tensor):
        end_point_pairs=batch_extend_lines_to_box(pairs=point_pairs,lower=self.lower,upper=self.upper)
        if self.reference=="end":
            new_pairs=end_point_pairs
        else:
            new_pairs=point_pairs#(m,2,d)抽样点本身，下文都以new pairs代替。
        #生成x点抽样池，在参考点之间随机抽取n个点
        m,_,d=new_pairs.shape
        start=new_pairs[:,0,:]#(m,d)
        end=new_pairs[:,1,:]#(m,d)
        # 随机生成权重 t ∈ [0,1]，shape: (m, k, 1)
        #t = torch.rand(m, self.num_interpolation_per_pair, 1, device=new_pairs.device, dtype=new_pairs.dtype)
        # 均匀生成t: (m, k, 1)，均匀分布于 [0,1]
        t = torch.linspace(0, 1, steps=self.num_interpolation_per_pair, device=new_pairs.device, dtype=new_pairs.dtype)
        t = t.view(1, -1, 1).expand(m, -1, -1)#(m, k, 1)
        # 插值：start.unsqueeze(1) shape → (m,1,d)
        pseudo_samples = (1 - t) * start.unsqueeze(1) + t * end.unsqueeze(1)  # (m, k, d)

        if self.w_adaptive:
            #solve w and b so that wx1+b=anchor_1, wx2+b=anchor_2
            w = start-end #(m,d)求差
            #print(w.shape)
            norm_sq = (w ** 2).sum(dim=1, keepdim=True)  # (m, 1)
            w = (self.s2-self.s1)* (w / norm_sq)  #(m,d)
            b = -(w*end).sum(dim=1,keepdim=True)+self.s1 #-wTx1+s1 (m,1)+(1,)=(m,1)
            
            return w,b,end_point_pairs,pseudo_samples #(m,1)*(m,d)->(m,d), (m,1) (m,2,d) (m,n,d)

        else:
            w = start-end #(m,d)求差
            w = torch.nn.functional.normalize(w,dim=1)#归一化
            w = self.w_scale*w #(m,d)乘上指定的固定系数
            middle_points = new_pairs.mean(dim=1)
            b = -torch.sum(w*middle_points,dim=1,keepdim=True) #(m,1)保证中点为0的b计算方法
            return w,b,end_point_pairs, pseudo_samples #(m,d) (m,1) (m,2,d) (m,k,d)
            


