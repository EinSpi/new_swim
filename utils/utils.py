import torch
from activations.activations import Activation
import gc
from scipy.spatial import KDTree

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






def find_nearest_indices(x_inter: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """
    对 x_inter (m,k,d) 中每个点，查找离 x_real (N,d) 中最近点的索引。

    返回:
        indices: (m, k, 1) 的 long 类型 Tensor，表示最近邻在 x_real 中的索引
    """
    m, k, d = x_inter.shape

    # 转 numpy，并展平为 (m*k, d)
    inter_flat = x_inter.reshape(-1, d).cpu().numpy()
    real_np = x_real.cpu().numpy()

    # 用 KDTree 查找最近邻索引
    tree = KDTree(real_np)
    _, indices = tree.query(inter_flat)  # indices: (m*k,)

    # 转回 torch，并 reshape 为 (m, k)
    indices_tensor = torch.tensor(indices, dtype=torch.long).reshape(m, k)

    return indices_tensor






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
                    a_params:torch.Tensor, activation: Activation)->torch.Tensor:
    #x:(T,d) w (N,d) b (N,1) a_params(N,p+q)
    num_neurons=w.shape[0]#N
    num_datas=x.shape[0]#T
    linear_transformed_x=w@x.T+b #(m,N)
    result= torch.zeros(num_neurons, num_datas, dtype=torch.float32)#empty (N,T)
    for i in range(num_neurons):
        
        result[i]=activation.infer(linear_transformed_x[i],None if a_params==None else a_params[i]).detach()
        
    return result.T#(T,N)



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

    


