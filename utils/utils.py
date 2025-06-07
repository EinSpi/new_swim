import numpy as np
import torch
from typing import Callable
from activations.activations import Activation
import gc
from scipy.spatial import KDTree

def generate_point_sets(x:torch.Tensor,y:torch.Tensor,size:int=2,number:int=1000,generator=None,random_seed=42)->tuple[torch.Tensor, torch.Tensor]:
    """
        从X, y中抽取m个batch，每个batch包含k个样本，PyTorch版。
        
        参数：
            X: Tensor, shape (N, dx)
            y: Tensor, shape (N, dy)
            m: int, batch数量
            k: int, 每个batch的样本数量
            random_seed: int, 随机种子（如果generator没提供）
            no_overlap_between_sets: bool, 是否要求不同batch之间也不重叠
            generator: torch.Generator实例（可选）
            
        返回：
            X_sampled: Tensor, shape (m, k, dx)
            y_sampled: Tensor, shape (m, k, dy)
        """
    
    if generator is None:
        generator = torch.Generator()
        if random_seed is not None:
            generator.manual_seed(random_seed)

    N= x.shape[0]
    N_y= y.shape[0]
    assert N == N_y, "X和y的第一个维度N必须一致"

    if size > N:
        raise ValueError(f"k={size}不能大于样本数量N={N}！")

    
    all_indices = torch.randint(0, N, size=(number, size), generator=generator)

    x_sampled = x[all_indices]  # (m, k, dx)
    y_sampled = y[all_indices]  # (m, k, dy)

    return x_sampled, y_sampled

def generate_x_point_pairs(x:torch.Tensor,number:int=1000,generator=None,random_seed=42)->torch.Tensor:
    if generator is None:
        generator = torch.Generator()
        if random_seed is not None:
            generator.manual_seed(random_seed)
    N= x.shape[0]
    all_indices = torch.randint(0, N, size=(number, 2), generator=generator)
    x_sampled = x[all_indices]
    #滤除离得过近的点对
    diffs = x_sampled[:, 0, :] - x_sampled[:, 1, :]  # (m, d)
    norms = torch.norm(diffs, dim=1)                 # (m,)
    keep_mask = norms >= 1e-4                  # (m,)
    x_sampled = x_sampled[keep_mask] 
    return x_sampled #(m-c,2,d)  c为不满足条件被过滤掉的点对个数


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

def batch_interpolate_between(endpoints: torch.Tensor, n: int) -> torch.Tensor:
    """
    对 m 对端点批量插值 n 个等距中间点（不含端点）

    参数:
        endpoints: (m, 2, d) - 每行两个端点
        n: int - 插值点数量

    返回:
        interpolated: (m, n, d)
    """
    p1 = endpoints[:, 0, :]  # (m, d)
    p2 = endpoints[:, 1, :]  # (m, d)
    direction = p2 - p1      # (m, d)

    # 生成比例系数 (n,) → (1, n, 1) for broadcasting
    alphas = torch.linspace(1, n, steps=n, device=p1.device, dtype=p1.dtype) / (n + 1)
    alphas = alphas.view(1, n, 1)  # shape: (1, n, 1)

    # 插值计算：p1.unsqueeze(1) → (m, 1, d)
    interpolated = p1.unsqueeze(1) + alphas * direction.unsqueeze(1)  # (m, n, d)
    return interpolated

def find_nearest_indices(x_inter: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """
    对 x_inter (m,k,d) 中每个点，查找离 x_real (N,d) 中最近点的索引。

    返回:
        indices: (m, k, 1) 的 long 类型 Tensor，表示最近邻在 x_real 中的索引
    """
    m, k, d = x_inter.shape
    N = x_real.shape[0]

    # 转 numpy，并展平为 (m*k, d)
    inter_flat = x_inter.reshape(-1, d).cpu().numpy()
    real_np = x_real.cpu().numpy()

    # 用 KDTree 查找最近邻索引
    tree = KDTree(real_np)
    distances, indices = tree.query(inter_flat)  # indices: (m*k,)

    # 转回 torch，并 reshape 为 (m, k)
    indices_tensor = torch.tensor(indices, dtype=torch.long).reshape(m, k)

    return indices_tensor


def find_principal_directions(X_batch: torch.Tensor) -> torch.Tensor:
    """
    批量找出主方向（第一主成分方向）。

    参数：
        X_batch: Tensor, shape (m, N, d)

    返回：
        directions: Tensor, shape (m, d)，每组一个单位向量
    """
    m, N, d = X_batch.shape

    # 中心化，每组分别减自己的均值
    mean = X_batch.mean(dim=1, keepdim=True)  # (m, 1, d)
    X_centered = X_batch - mean               # (m, N, d)

    # 批量SVD
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)  # Vh: (m, d, d)

    # 每组的第一个右奇异向量 (主方向)
    principal_directions = Vh[:, 0, :]  # (m, d)

    # 单位化（虽然SVD出来基本上是单位模长，但保险再正规化一次）
    principal_directions = torch.nn.functional.normalize(principal_directions, dim=1)

    return principal_directions

def w_b_calculator(candidate_x_pool:torch.Tensor)-> tuple[torch.Tensor, torch.Tensor]:
        #calculate the weights and biases from self.candidate_pool
        candidate_w_pool = find_principal_directions(candidate_x_pool) #(m,d)
        candidate_x_center_pool=candidate_x_pool.mean(dim=1)#(m,d)
        candidate_b_pool = -torch.sum(candidate_w_pool*candidate_x_center_pool,dim=1,keepdim=True) #(m,1)
        return candidate_w_pool, candidate_b_pool

def clean_inputs(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将输入x, y处理成规范形状。

    - x: 保证shape是 (N, -1)
    - y: 保证shape是 (N, 1)

    参数:
        x (torch.Tensor): 输入张量，任意shape
        y (torch.Tensor): 标签张量，任意shape

    返回:
        tuple (x_cleaned, y_cleaned)
    """
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if y.ndim < 2:
        y = y.reshape(-1, 1)
    return x, y
    
def general_forward(x:torch.Tensor, w: torch.Tensor, b:torch.Tensor, 
                    a_params:torch.Tensor,poles:torch.Tensor, activation: Activation)->torch.Tensor:
    #x:(N,d) w (m,d) b (m,1) a_params(m,p)
    num_neurons=w.shape[0]#m
    num_datas=x.shape[0]#N
    linear_transformed_x=w@x.T+b #(m,N)
    result= torch.zeros(num_neurons, num_datas, dtype=torch.float32)#empty (m,N)
    for i in range(num_neurons):
        
        result[i]=activation.infer(linear_transformed_x[i],None if a_params==None else a_params[i],None if poles==None else poles[i]).detach()
        
    return result.T#(N,m)

def column_normalize_and_append_1(a:torch.Tensor):
    #a(N,m)
    num_rows=a.shape[0]
    target_norm=torch.sqrt(torch.tensor(num_rows,dtype=a.dtype,device=a.device)) #(1,)
    # 原始每列的 L2 范数 (m,)
    col_norms = torch.norm(a, dim=0, keepdim=True)  # shape: (1, m)
    #缩放系数
    scale = target_norm / (col_norms + 1e-8) #(1,m)
    #缩放每一列
    a_scaled = a*scale#(N,m)*(1,m)
    #添加一列全1
    ones_col = torch.ones((num_rows, 1), dtype=a.dtype, device=a.device)
    #拼接
    a_aug = torch.cat([a_scaled, ones_col], dim=1)  # shape: (N, m+1)
    return a_aug

def probability_calculator(inte_pool_forward:torch.Tensor,y:torch.Tensor,full_matrices:bool=False)->torch.Tensor:
    """
    #coefficients idea
    #print(inte_pool_forward.dtype)
    _,_,Vh=torch.linalg.svd(torch.cat([inte_pool_forward, y], dim=1),full_matrices=True)
    #w,*_=torch.linalg.lstsq(inte_pool_forward, y)
    #w=w[:-1].squeeze(-1)
    v=Vh[-1]
    v = v / (-v[-1])
    #print(v.shape)
    w = v[:-2]
    #print(w.shape)
    print(w.shape)
    prob = torch.abs(w)/torch.sum(torch.abs(w))
    """

    ##cosine based probability
    squared_inner_products=(y.T@inte_pool_forward)**2 
    print("inner ")
    print(squared_inner_products.shape)
    norms=torch.sum(inte_pool_forward**2,dim=0,keepdim=True)
    print("norms ")
    print(norms.shape)
    logits=(squared_inner_products/norms).squeeze()
    prob=logits/torch.sum(logits)
    return prob 

def delete_tensors(*tensors, clear_cuda_cache: bool = True):
    """
    删除任意数量的 Tensor，并可选清除 GPU 显存缓存。
    
    参数：
        *tensors: 任意数量的 Tensor 变量（可以是列表、元组、变量名）
        clear_cuda_cache: 是否自动清理 CUDA 缓存，默认 True
    """
    for t in tensors:
        del t
    gc.collect()  # 强制垃圾回收
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

def a_params_initializer(x_points,y_targets,num_coeff_p,num_coeff_q):
    unknown_num=num_coeff_p+num_coeff_q#p+q
    indices = torch.randperm(x_points.shape[0])[:unknown_num]  # 生成不重复随机索引
    selected_x_points=x_points[indices]#(p+q,)
    selected_y_targets=y_targets[indices]#(p+q,)
    numerator_matrix=torch.stack([selected_x_points ** i for i in range(num_coeff_p)], dim=1) #(p+q, p)
    denominator_matrix=torch.stack([(selected_x_points ** (2*i)) for i in range(num_coeff_q)],dim=1)*(-selected_y_targets.unsqueeze(-1))#(p+q,q)*(p+q,1)=(p+q,q)
    coefficient_matrix=torch.cat([numerator_matrix,denominator_matrix],dim=1)#(p+q,p+q)
    _, _, Vh = torch.linalg.svd(coefficient_matrix,full_matrices=True)#(p+q,p+q)
    v = Vh[-1]
    return v.clone().detach().requires_grad_()
    


