from dataclasses import dataclass
import torch
def normalize_and_append_1(F:torch.Tensor):
    """
    返回F统一模长且添加一列1。

        Args:
            F (torch.Tensor): 待处理矩阵 (T,N)

        Returns:
            result (torch.Tensor): 处理后的统一模长矩阵 (T,N+1)
    """
    T, N = F.shape
    target_norm = T ** 0.5

    # Step 1: 计算每一列的模长（2范数）
    col_norms = torch.norm(F, p=2, dim=0, keepdim=True)+1e-8 # shape: (1, N)

    # Step 2: 缩放每列使其模长为 sqrt(T)
    normalized_F = F / col_norms * target_norm

    # Step 3: 创建一列全1，shape为 (T, 1)
    ones_column = torch.ones((T, 1), dtype=F.dtype, device=F.device)

    # Step 4: 拼接
    return torch.cat([normalized_F, ones_column], dim=1)

def p1(x_pairs:torch.Tensor,y_points:torch.Tensor)->torch.Tensor:
    """
    返回方差指导的概率分布。

    Args:
        x_pairs (torch.Tensor): 随机抽取的x点对 (N,2,d)
        y_points (torch.Tensor): y插值点 (N,k,1)

    Returns:
        prob (torch.Tensor): 概率分布 (N,)
    """
    var=torch.var(y_points, dim=1, unbiased=False).squeeze(-1) # (N,)
    #diff = x_pairs[:, 0, :] - x_pairs[:, 1, :]     # shape: (N, d)
    #l2 = torch.norm(diff, dim=1, keepdim=False)  # shape: (N,)
    #l2 = l2/torch.max(l2) #(N,)
    #logits=var/l2
    logits=var
    prob=logits/torch.sum(logits) #(N,)
    return prob
     
def p2(y:torch.Tensor,F:torch.Tensor)->torch.Tensor:
    """
    返回cos值指导的概率分布。

    Args:
        y (torch.Tensor): 目标函数 (T,1)
        F (torch.Tensor): 所有激活函数 (T,N)

    Returns:
        prob (torch.Tensor): 概率分布 (N,)
    """
    F_prime=normalize_and_append_1(F=F)#(T,N+1)
    squared_inner_products=(y.T@F_prime)**2 
    norms=torch.sum(F_prime**2,dim=0,keepdim=True).clamp_min(1e-8)
    logits=(squared_inner_products/norms).squeeze()
    logits=logits[:-1]#去掉最后一个，返回宽度N
    prob=logits/torch.sum(logits)
    return prob
    
def p3(y:torch.Tensor,F:torch.Tensor)->torch.Tensor:
    """
    返回系数指导的概率分布。

    Args:
        y (torch.Tensor): 目标函数 (T,1)
        F (torch.Tensor): 所有激活函数 (T,N)

    Returns:
        prob (torch.Tensor): 概率分布 (N,)
    """
    F_prime=normalize_and_append_1(F=F)#(T,N+1)
    F_prime+=1e-6 * torch.randn_like(F_prime) #即将svd，加入随机噪声破坏奇异性
    _,_,Vh=torch.linalg.svd(torch.cat([F_prime, y], dim=1),full_matrices=True)
    v=Vh[-1]#(N+2)
    v = v / (-v[-1])#把最后一个调为-1
    w = (v[:-2])**2#截掉最后两个，还原N
    prob = w/torch.sum(w)
    return prob


def p4(y:torch.Tensor, F:torch.Tensor,max_epochs:int=1000)->torch.Tensor:

    M=torch.eye(F.shape[0], requires_grad=True,device=F.device) 
    optimizer = torch.optim.Adam([M], lr=0.001)
    # 早停设置
    patience = max_epochs//5
    best_loss = float('inf')
    best_epoch = 0
    best_M = None
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss = diag_loss(M, F)
        
        if loss.item() < best_loss:  
            best_loss = loss.item()
            best_epoch = epoch
            best_M = M.detach().clone()
        elif epoch - best_epoch > patience:
            print(f"Early stopping at epoch {epoch}. Best loss {best_loss:.6f} at epoch {best_epoch}")
            break

        loss.backward()
        optimizer.step()
        #if epoch%100==0:
            #print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    print(f"Best offdiag loss {best_loss:.6f} at epoch {best_epoch}")

    return p2(y=best_M@y,F=best_M@F)

def p5(y_points:torch.Tensor)->torch.Tensor:
     #uniform
     logits=torch.ones(y_points.shape[0],device=y_points.device)
     prob = logits/torch.sum(logits)
     return prob
        
def p6(x_pairs:torch.Tensor,y_points:torch.Tensor):
    """
    返回传统SWIM的概率分布。

    Args:
        x_pairs (torch.Tensor): 随机抽取的x点对 (N,2,d)
        y_pairs (torch.Tensor): y点对 (N,2,1)

    Returns:
        prob (torch.Tensor): 概率分布 (N,)
    """

    diff_y=torch.norm(y_points[:,0,:]- y_points[:,-1,:],dim=-1)#(N,)
    diff_x=torch.norm(x_pairs[:,0,:] - x_pairs[:,-1,:] ,dim=-1).clamp_min(1e-6)#(N,)
    logits = diff_y/diff_x#(N,)
    prob=logits/torch.sum(logits)
    return prob



def diag_loss(M:torch.Tensor, F:torch.Tensor):
    A = F.T @ M @ F  # (5, 5)
    off_diag = A - torch.diag(torch.diagonal(A))
    return torch.mean(off_diag**2)  # 非对角程度


@dataclass
class Probability_Solver:
    prob_strategy:str="var"
    max_epochs:int=1000
    def probability_calculator(self,tensor1:torch.Tensor,tensor2:torch.Tensor)->torch.Tensor:
         if self.prob_strategy=="var":
              return p1(x_pairs=tensor1,y_points=tensor2)
         elif self.prob_strategy=="cos":
              return p2(y=tensor1,F=tensor2)
         elif self.prob_strategy=="coeff":
              return p3(y=tensor1,F=tensor2)
         elif self.prob_strategy=="M":
              return p4(y=tensor1,F=tensor2,max_epochs=self.max_epochs)
         elif self.prob_strategy=="uni":    
              return p5(y_points=tensor2)
         elif self.prob_strategy=="SWIM":
              return p6(x_pairs=tensor1,y_points=tensor2)
         else:
              raise ValueError("undefined prob strategy")
         
        

