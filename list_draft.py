import itertools
import inspect

def iterate_lists(*lists,exp):
    # 获取调用者的局部变量名（用于找到变量名）
    caller_locals = inspect.currentframe().f_back.f_locals

    # 查出传入列表对应的变量名（逆查）
    var_names = []
    for lst in lists:
        found = [name for name, val in caller_locals.items() if val is lst]
        var_names.append(found[0] if found else 'unknown')

    # 输出：去掉 s 的变量名
    stripped_names = [name.rstrip('s') for name in var_names]
    print("Variable names (singular form):", ','.join(stripped_names))

    # 分离前 n-1 和最后一个列表
    *leading_lists, last_list = lists

    for combo in itertools.product(*leading_lists):
        for item in last_list:
            cmd="python3 SWIM.py  --path_keys "+','.join(stripped_names)+" --device 0 --exp "+exp+" --"+stripped_names[-1]+f" {item}"
            for i in range(len(combo)):
                cmd=cmd+ " --"+stripped_names[i]+f" {combo[i]} "
            print(cmd)
            #print(last_list)

objs = ['KdV_sine','discontinuous_complicated']
acts = ['rat']
widths=[25,50,100,200,400,800,1600]

iterate_lists(objs,acts,widths,exp="exp_0620")
