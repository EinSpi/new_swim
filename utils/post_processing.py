import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy.interpolate import griddata
from utils.plotting import newfig, savefig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
def compute_mse(tensor1,tensor2):
    return np.mean((tensor1-tensor2)**2)

def compute_rel_l2(tensor1,tensor2):
    return np.linalg.norm(tensor1-tensor2, ord=2) / np.linalg.norm(tensor2, ord=2)

def save_errors(save_path: str, mse: float, rel_l2: float,):
    os.makedirs(save_path, exist_ok=True)
    mse_error_file = os.path.join(save_path, 'mse_errors.txt')
    rel_error_file = os.path.join(save_path, 'rel_errors.txt')

    mse_index = get_line_index(mse_error_file)
    rel_index = get_line_index(rel_error_file)
    
    with open(mse_error_file, 'a') as f:
        f.write(f"{mse_index}: mse = {mse}\n")
    with open(rel_error_file, 'a') as f:
        f.write(f"{rel_index}: mse = {rel_l2}\n")

def plot_dynamics(Exact_idn,lb_idn,ub_idn,keep,U_pred,save_path):
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    fig, ax = newfig(1.0, 0.6)
    ax.axis('off')

    ######## Exact solution #######################
    ########      Predicted p(t,x,y)     ###########
    gs = gridspec.GridSpec(1, 3)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs[:, 0])
    h = ax.imshow(Exact_idn, interpolation='nearest', cmap='jet',
                    extent=[lb_idn[0], ub_idn[0]*keep, lb_idn[1], ub_idn[1]],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Exact Dynamics', fontsize = 10)

    ######## Approximate ###########
    ax = plt.subplot(gs[:, 1])
    h = ax.imshow(U_pred, interpolation='nearest', cmap='jet',
                    extent=[lb_idn[0], ub_idn[0]*keep, lb_idn[1], ub_idn[1]],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Predicted' , fontsize = 10)

    ######## Approximation Error ###########
    ax = plt.subplot(gs[:, 2])
    h = ax.imshow(np.abs(Exact_idn-U_pred), interpolation='nearest', cmap='jet',
                    extent=[lb_idn[0], ub_idn[0] * keep, lb_idn[1], ub_idn[1]],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Error', fontsize=10)

    # 获取文件夹中已有的文件数（只计数文件）
    dynamics_path=os.path.join(save_path, "dynamics")
    os.makedirs(dynamics_path, exist_ok=True)
    existing_files = [f for f in os.listdir(dynamics_path)]
    idx = len(existing_files)
    dynamic_path = os.path.join(dynamics_path, f"dynamic{idx}")
    savefig(dynamic_path)

    # 获取当前行数作为索引
def get_line_index(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

    
    
    
