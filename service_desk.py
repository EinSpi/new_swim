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
                  args_combo:Tuple, last_arg, stripped_var_names:list[str], 
                  device, cwd, queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    cmd="python3 SWIM.py  --path_keys "+','.join(stripped_var_names)+" --device 0 --exp "+exp+" --"+stripped_var_names[-1]+f" {last_arg}"
    for i in range(len(args_combo)):
        cmd=cmd+ " --"+stripped_var_names[i]+f" {args_combo[i]} "
    
    subprocess.call(cmd, shell=True, cwd=cwd,env=env)
    queue.put(device)


    

def iterate_lists(*lists,exp,cwd):
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
            devices = get_i_best_gpus(2)
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

exp = 'exp_flexible_args'
objs = ['KdV_sine','discontinuous_complicated']
acts = ['rat']
widths=[25,50,100,200,400,800,1600]

iterate_lists(objs,acts,widths,exp=exp,cwd=cwd)

"""
with tqdm.tqdm(total=total, desc="Progress", unit="task", colour="green") as pbar:
    for width in widths:
        for act in acts:
            devices = get_i_best_gpus(2)
            queue = Queue()
            running = {}  # gpu_id: Process
            remaining_objs = objs.copy()
            # Pre-fill the queue with initial free GPUs
            for device in devices:
                queue.put(device)
            while remaining_objs:
                if not queue.empty():
                        device = queue.get()
                        pbar.update(1)
                        if device in running and running[device].is_alive():
                            running[device].join()
                        obj = remaining_objs.pop(0)
                        p = Process(target=run_training, args=(exp,obj,act,width,device,cwd,queue))
                        p.start()
                        running[device] = p
                else:
                    # If no GPU reports finished yet, wait a bit
                    time.sleep(1)
            
            # Wait for all running processes to finish
            for p in running.values():
                p.join()



"""
    



