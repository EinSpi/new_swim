import torch
import scipy.io
import numpy as np

def load_training_data_from_mat(
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
