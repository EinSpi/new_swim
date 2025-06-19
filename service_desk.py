import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
import tqdm
import numpy as np
import tensorflow as tf
import time
from multiprocessing import Process, Queue

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
    
    cmd = f'python SWIM.py --exp {exp} --obj {obj} --act {act} --width {width} --device {device}'

    subprocess.call(cmd, shell=True, cwd=cwd)
    queue.put(device)  # Notify completion


cwd = os.path.dirname(os.path.realpath(__file__))

exp = 'exp_0620'
objs = ['KdV_sine','discontinuous_complicated']
acts = ['rat']
widths=[25,50,100,200,400,800,1600]

total=len(objs)*len(objs)*len(widths)

with tqdm.tqdm(total=total, desc="Progress", unit="task", colour="green") as pbar:
    for width in widths:
        for act in acts:
            devices = get_i_best_gpus(3)
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




    



