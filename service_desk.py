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

def run_training2(exp:str, path_keys:list, args_combo:Tuple,
                  dict2:dict,last_arg,
                  device, cwd, queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    cmd="python3 SWIM.py --exp "+exp
    #要统计的,这里差一个，因为args_combo比path_keys短一个
    for i in range(len(args_combo)):
        cmd=cmd+ f" --{path_keys[i]} {args_combo[i]} "
    cmd=cmd+f" --{path_keys[-1]} {last_arg} "
    #不要统计的
    for key, value in dict2.items():
        cmd=cmd+ f" --{key} {value} "
    cmd=cmd+" --device 0 --path_keys "+' '.join(path_keys)
    
    subprocess.call(cmd, shell=True, cwd=cwd,env=env)
    queue.put(device)


    

def iterate_lists(dict1,dict2,exp,cwd,gpu_num):
    keys1=list(dict1.keys())#需要统计的项目名称列表，随后要传入path_keys

    # 拆出前 n-1 和最后一个键
    leading_keys1, last_key1 = keys1[:-1], keys1[-1]
    leading_lists1 = [dict1[k] for k in leading_keys1]
    last_list1 = dict1[last_key1]

    total=1
    for lst in dict1.values():
        total=total*len(lst)

    # 生成组合
    with tqdm.tqdm(total=total, desc="Progress", unit="task", colour="green") as pbar:
        for combo in itertools.product(*leading_lists1):
            devices = get_i_best_gpus(gpu_num)
            queue = Queue()
            running = {}  # gpu_id: Process
            remaining_items = last_list1.copy()
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
                        p = Process(target=run_training2, args=(exp,keys1,combo,dict2,last_arg,device,cwd,queue))
                        p.start()
                        running[device] = p
                else:
                    # If no GPU reports finished yet, wait a bit
                    time.sleep(1)
                # Wait for all running processes to finish
            for p in running.values():
                p.join()
        




#######################################################################################################
cwd = os.path.dirname(os.path.realpath(__file__))
gpu_num=2
exp = 'exp_0620'


###你想统计的所有实验超参列表，注意名称必须加s，可以是一个，必须名称符合SWIM.py的参数名称

dict1={
    "obj":['KdV_sine','discontinuous_complicated','advection','discontinuous_trivial','burgers', 'euler_bernoulli'],
    "act":['rat'],
    "width":[50,100,200,400,800,1600],
    "loss_metric":["cos","mse"],
    "prob_strat":["var","cos","coeff","M"],

}
###你不想统计的但想修改的超参列表，必须没有上面项目，以词典的形式,必须是一个值
dict2={
    "reg_factor":1e-6,
    "max_epoch":3000,
    "random_seed":76
}

overlap=set(dict1.keys())&set(dict2.keys())
if set(dict1.keys())&set(dict2.keys()):
    raise ValueError(f"same keys in dict1 and dict2: {overlap}")

iterate_lists(dict1,dict2,exp=exp,cwd=cwd,gpu_num=gpu_num)


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