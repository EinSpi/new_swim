import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
import tqdm
import numpy as np
import time
from multiprocessing import Process, Queue
import itertools
import argparse
import json

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
    
def run_training(exp:str, total_dict:dict, path_keys:list,device:int, cwd:str, device_queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    cmd=f"python3 SWIM_train.py --exp {exp} "
    cmd+=" ".join([f"--{key} {value} " for key,value in total_dict.items()])
    cmd+=" --device 0 --path_keys "+' '.join(path_keys)
    
    subprocess.call(cmd, shell=True, cwd=cwd,env=env)
    device_queue.put(device)

def launch_parallel_experiment(dict1:dict,dict2:dict,exp:str,cwd:str,gpu_num:int):
    # 获取所有待统计的超参组合
    path_keys = list(dict1.keys())
    values = [dict1[key] for key in path_keys]
    all_value_combos_dict1 = list(itertools.product(*values))  # 全组合
    total = len(all_value_combos_dict1)

    with tqdm.tqdm(total=total, desc="Progress", unit="task", colour="green") as pbar:
        task_queue = all_value_combos_dict1.copy()
        device_queue = Queue()
        running = {}  # gpu_id -> Process

        # 初始填充 GPU 队列
        available_devices = get_i_best_gpus(gpu_num)
        for dev in available_devices:
            device_queue.put(dev)

        while task_queue:
            # 若 GPU 空闲且有任务，派发任务
            if not device_queue.empty():
                device = device_queue.get()
                value_combo_dict1 = task_queue.pop(0)
                combo_dict1=dict(zip(path_keys,value_combo_dict1))
                total_dict={**dict2, **combo_dict1}
                p = Process(target=run_training, args=(exp, total_dict, path_keys,device, cwd, device_queue))
                p.start()
                running[device] = p
                pbar.update(1)
            else:
                # 等 GPU 空闲
                time.sleep(1)

        # 等所有任务结束
        for p in running.values():
            p.join()



   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="命令行参数示例")
    
    # 添加参数
    parser.add_argument("--exp_config", type=str, default=" ", help="实验配置文件")
    args = parser.parse_args()
    
    #加载实验配置文件
    with open(args.exp_config, "r") as f:
        config = json.load(f)

    #锚定主操作目录，整理实验配置
    cwd :str= os.path.dirname(os.path.realpath(__file__))
    gpu_num:int=config["gpu_num"]
    exp:str=config["exp"]
    dict1:dict=config["dict1"]
    dict2:dict=config["dict2"]

    #冲突检查门禁
    overlap=set(dict1.keys())&set(dict2.keys())
    if set(dict1.keys())&set(dict2.keys()):
        raise ValueError(f"same keys in dict1 and dict2: {overlap}")
    
    #启动！
    launch_parallel_experiment(dict1,dict2,exp=exp,cwd=cwd,gpu_num=gpu_num)

    
    
"""
parser.add_argument("--exp", type=str, default="618", help="实验名称")
parser.add_argument("--obj", type=str, default="KdV_sine", help="目标函数")
parser.add_argument("--act",type=str, default="rat",help="激活函数种类")
parser.add_argument("--width", type=int, default=600, help="网络宽度")
parser.add_argument("--rep_scaler",type=int,default=2,help="抽样池尺寸是需要神经元数的几倍")
parser.add_argument("--loss_metric", type=str,default="mse", help="拟合adaptive参数时用的优化目标")
parser.add_argument("--prob_strat",type=str,default="var",help="用何种标准计算概率")
parser.add_argument("--init_method",type=str,default="relu_like",help="怎样初始化a_params,如果有")
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