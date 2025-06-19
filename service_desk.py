import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
import tqdm
import numpy as np
import time
from multiprocessing import Process, Queue
from typing import Tuple
import itertools
import inspect

def get_i_best_gpus(i):
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_usages = [int(x) for x in result.strip().split('\n')]
        best_i = np.argsort(memory_usages)[:i].tolist()
        return best_i
    except Exception as e:
        print("[WARNING] Failed to auto-select GPUs, fallback to 0 and 1.")
        print(e)
        return [0, 1]
    
def run_training(exp, obj, act, width, device,cwd,queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    
    cmd = f'python3 SWIM.py --exp {exp} --obj {obj} --act {act} --width {width} --device 0'

    subprocess.call(cmd, shell=True, cwd=cwd,env=env)
    queue.put(device)  # Notify completion

def run_training2(exp:str, 
                  args_combo:Tuple, last_arg, stripped_var_names:list, 
                  device, cwd, queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    cmd="python3 SWIM.py --exp "+exp
    for i in range(len(args_combo)):
        cmd=cmd+ " --"+stripped_var_names[i]+f" {args_combo[i]} "
    cmd=cmd+" --"+stripped_var_names[-1]+f" {last_arg}"+" --device 0 --path_keys "+','.join(stripped_var_names)
    subprocess.call(cmd, shell=True, cwd=cwd,env=env)
    queue.put(device)


    

def iterate_lists(*lists,exp,cwd,gpu_num):
    #获取总任务数
    total=1
    for lst in lists:
        total=total*len(lst)
     # 获取调用者的局部变量名（用于找到变量名）
    caller_locals = inspect.currentframe().f_back.f_locals
    # 查出传入列表对应的变量名（逆查）
    var_names = []
    for lst in lists:
        found = [name for name, val in caller_locals.items() if val is lst]
        var_names.append(found[0] if found else 'unknown')

    # 输出：去掉 s 的变量名
    stripped_names = [name.rstrip('s') for name in var_names]
    
    # 分离前 n-1 和最后一个列表
    *leading_lists, last_list = lists


    with tqdm.tqdm(total=total, desc="Progress", unit="task", colour="green") as pbar:
        for combo in itertools.product(*leading_lists):
            devices = get_i_best_gpus(gpu_num)
            queue = Queue()
            running = {}  # gpu_id: Process
            remaining_items = last_list.copy()
            # Pre-fill the queue with initial free GPUs
            for device in devices:
                queue.put(device)
            while remaining_items:
                if not queue.empty():
                        device = queue.get()
                        pbar.update(1)
                        if device in running and running[device].is_alive():
                            running[device].join()
                        last_arg = remaining_items.pop(0)
                        p = Process(target=run_training2, args=(exp,combo,last_arg,stripped_names,device,cwd,queue))
                        p.start()
                        running[device] = p
                else:
                    # If no GPU reports finished yet, wait a bit
                    time.sleep(1)
            
            # Wait for all running processes to finish
            for p in running.values():
                p.join()


        


cwd = os.path.dirname(os.path.realpath(__file__))
gpu_num=2
exp = 'exp_flexible_args'


###你想统计的所有实验超参列表，注意名称必须加s，必须名称符合SWIM.py的参数名称
objs = ['KdV_sine','discontinuous_complicated']
acts = ['rat']
widths=[100,200]
rep_scalers=[2,4]
loss_metrics=["cos", "mse"]
prob_strat=["var","cos","coeff","M"]
optimizers=["adam"]
set_sizes=[7,8]


iterate_lists(objs,acts,widths,rep_scalers,loss_metrics,prob_strat,optimizers,set_sizes,exp=exp,cwd=cwd,gpu_num=gpu_num)


"""
parser.add_argument("--exp", type=str, default="618", help="实验名称")
parser.add_argument("--obj", type=str, default="KdV_sine", help="目标函数")
parser.add_argument("--act",type=str, default="rat",help="激活函数种类")
parser.add_argument("--width", type=int, default=600, help="网络宽度")
parser.add_argument("--rep_scaler",type=int,default=2,help="抽样池尺寸是需要神经元数的几倍")
parser.add_argument("--loss_metric", type=str,default="mse", help="拟合adaptive参数时用的优化目标")
parser.add_argument("--prob_strat",type=str,default="var",help="用何种标准计算概率")
parser.add_argument("--optimizer", type=str, default="adam", help="子优化任务优化器")
parser.add_argument("--p", type=int,default=4,help="有理函数的分子阶数")
parser.add_argument("--q",type=int,default=3,help="有理函数的分母阶数")
parser.add_argument("--set_size", type=int,default=7,help="点集大小")

parser.add_argument("--max_epoch",type=int,default=3000,help="局部优化adaptive parameter最大迭代次数")
parser.add_argument("--M_max_epoch",type=int,default=500,help="优化M矩阵最大迭代次数")
parser.add_argument("--reg_factor",type=float,default=1e-6,help="第二个归一化系数")

parser.add_argument("--sample_first",type=bool,default=True,help="先抽样还是先拟合")
parser.add_argument("--random_seed", type=int,default=92,help="随机种子")
parser.add_argument("--int_sketch",type=bool, default=True,help="是否要绘制中间激活函数图像")
parser.add_argument("--save_weight", type=bool,default=True, help="是否要保存训练后的权重")
parser.add_argument("--device",type=int, default=0, help="实验要使用的设备编号") 
"""