from dataclasses import dataclass
import torch

@dataclass
class Pole_Supression:
    pole_policy:str="has_poles"
    pole_out_shift_factor:float=3.0
    def compute_poles(self, w:torch.Tensor, b:torch.Tensor, end_point_pairs:torch.Tensor):
        if self.pole_policy=="no_poles":
            return None
        else:
            #w(m,d) b(m,1) end_point_pairs (m,2,d)
            end_point_pairs_1d=torch.bmm(end_point_pairs, w.unsqueeze(2))+b.unsqueeze(2) #(m,2,1)+(m,1,1)=(m,2,1)
            poles=self.pole_out_shift_factor*end_point_pairs_1d.squeeze(-1)#(m,2) 
            return poles #(m,2)
    

