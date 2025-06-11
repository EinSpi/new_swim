import torch
import scipy.io
import numpy as np
import activations.activations as act
import swim_backbones.dense
import swim_backbones.linear
import swim_model

def load_data(
                                data_path: str ,
                                N_train: int = 10**4,
                                N_val: int = 10**4,
                                noise: float = 0.0,
                                seed: int = None):
    """
    从 .mat 文件读取 (t, x, u) 数据，并构造训练集和验证集（PyTorch 版本）
    """
    if seed is not None:
        np.random.seed(seed)

    # Load .mat
    data_idn = scipy.io.loadmat(data_path)

    t_idn = data_idn['t'].flatten()[:, None]     # (Nt, 1)
    x_idn = data_idn['x'].flatten()[:, None]     # (Nx, 1)
    Exact_idn = np.real(data_idn['usol'])         # (Nx, Nt)

    # Meshgrid
    T_idn, X_idn = np.meshgrid(t_idn, x_idn)                 # both (Nx, Nt)

    # Flatten to (Nx*Nt, 1)
    t_idn_star = T_idn.flatten()[:, None]
    x_idn_star = X_idn.flatten()[:, None]
    X_idn_star = np.hstack((t_idn_star, x_idn_star))
    u_idn_star = Exact_idn.flatten()[:, None]

    # Random sampling for train/val
    
    idx = np.random.choice(t_idn_star.shape[0], N_train + N_val, replace=False)
    idx_train = idx[:N_train]
    idx_val = idx[N_train:]

    t_train = t_idn_star[idx_train]
    x_train = x_idn_star[idx_train]
    u_train = u_idn_star[idx_train]

    t_val = t_idn_star[idx_val]
    x_val = x_idn_star[idx_val]
    u_val = u_idn_star[idx_val]

    # Add noise to training targets
    u_train_noisy = u_train + noise * np.std(u_train) * np.random.randn(*u_train.shape)

    # Stack into (N, 2)
    X_train_np = np.hstack([t_train, x_train])
    X_val_np = np.hstack([t_val, x_val])

    # Convert to torch
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    U_train = torch.tensor(u_train_noisy, dtype=torch.float32)

    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    U_val = torch.tensor(u_val, dtype=torch.float32)

    X_idn_star = torch.tensor(X_idn_star,dtype=torch.float32)

    return X_train, U_train, X_val, U_val, X_idn_star, u_idn_star, T_idn, X_idn, Exact_idn

def activation_prepare(activation:str="rat",p:int=4,q:int=3):
    if activation=="rat":
        return act.Rational(num_coeff_p=p,num_coeff_q=q)
    elif activation=="adpt_tanh":
        return act.Adpt_Tanh()
    elif activation=="adpt_relu":
        return act.Adpt_Relu()
    elif activation=="adpt_sigmoid":
        return act.Adpt_Sigmoid()
    elif activation=="tanh":
        return act.Tanh()
    elif activation=="relu":
        return act.Relu()
    elif activation=="sigmoid":
        return act.Sigmoid()

def model_prepare(activation:act.Activation=act.Tanh(),
                  layer_width:int=200,random_seed:int=42,
                  repetition_scaler:int=4,set_size:int=7,loss_metric:str="mse",prob_strategy:str="cos",
                  reg_factor_1:float=0.0,reg_factor_2:float=1e-6,int_sketch:bool=True,save_path:str=" ")->swim_model.Swim_Model:
    dense=swim_backbones.dense.Dense(activation=activation,
                                     layer_width=layer_width,random_seed=random_seed,
                                     repetition_scaler=repetition_scaler,set_size=set_size,loss_metric=loss_metric,prob_strategy=prob_strategy,
                                    reg_factor_1=reg_factor_1,reg_factor_2=reg_factor_2,int_sketch=int_sketch,save_path=save_path
                                    )
    linear=swim_backbones.linear.Linear()
    model=swim_model.Swim_Model([dense,linear])
    return model