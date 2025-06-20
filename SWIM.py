import os
import argparse
import activations.activations as act
import utils.pre_processing as prp
import numpy as np
import torch
from w_b_solver.w_b_solver import W_B_Solver
from probability_solver.probability_solver import Probability_Solver
from adaptive_solver.adaptive_solver import Adaptive_Solver
import SWIM_test

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="命令行参数示例")

    # 添加参数
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

    parser.add_argument("--max_epochs",type=int,default=3000,help="局部优化adaptive parameter最大迭代次数")
    parser.add_argument("--M_max_epochs",type=int,default=500,help="优化M矩阵最大迭代次数")
    parser.add_argument("--reg_factor",type=float,default=1e-6,help="第二个归一化系数")
    
    parser.add_argument("--sample_first",type=bool,default=True,help="先抽样还是先拟合")
    parser.add_argument("--random_seed", type=int,default=92,help="随机种子")
    parser.add_argument("--int_sketch",type=bool, default=True,help="是否要绘制中间激活函数图像")
    parser.add_argument("--save_weights", type=bool,default=True, help="是否要保存训练后的权重")
    parser.add_argument("--device",type=int, default=0, help="实验要使用的设备编号") 

    parser.add_argument("--path_keys", type=str, default="exp,obj,act,width",help="哪些参数参与实验统计，用逗号分隔")
    
    args = parser.parse_args()

    #########################################
    ############目录准备阶段##################
    ########################################
    #创建实验目录：
    #要参加统计的项目列表
    include_arguments_list=args.path_keys.split(",")
    #所有参数列表
    all_arg_names = [action.dest for action in parser._actions if action.option_strings and hasattr(args, action.dest)]
    #所有参数去掉最后一个 path_keys
    if "path_keys" in all_arg_names:
        all_arg_names.remove("path_keys")
    #组建实验目录
    experiment_path = "Results"
    for name in all_arg_names:
        if name in include_arguments_list:
            value = getattr(args, name)
            experiment_path += f"/{name}_{value}"

    os.makedirs(experiment_path, exist_ok=True)
    #在实验目录中写下所有参数到一个文件中，作为manifest
    manifest_path = os.path.join(experiment_path, "manifest.txt")
    with open(manifest_path, "w") as f:
        for name in all_arg_names:
            value = getattr(args, name)
            f.write(f"{name}={value}\n")

    ###################################################
    #################训练阶段###########################
    ###################################################
    
    #准备训练数据
    X_train, u_train, utrain_numpy,X_val, u_val, X_idn_star, u_idn_star, T_idn, X_idn, Exact_idn=prp.load_data(data_path="Data/"+args.obj+".mat", seed=args.random_seed)
    #准备激活函数
    activation=prp.activation_prepare(activation=args.act,p=args.p,q=args.q)
    #准备模型
    ##准备设备
    device= torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    ##准备随机数生成器
    gpu_gen=torch.Generator(device=device)
    gpu_gen.manual_seed(args.random_seed)
    cpu_gen=torch.Generator()
    cpu_gen.manual_seed(args.random_seed)
    ##准备三个求解器，权重，a_para,概率
    w_b_solver = W_B_Solver(num_interpolation_per_pair=args.set_size)
    adaptive_solver=Adaptive_Solver(loss_metric=args.loss_metric,
                                    reg_factor=args.reg_factor,
                                    int_sketch=args.int_sketch,
                                    optimizer=args.optimizer,
                                    cpu_gen=cpu_gen,
                                    save_path=experiment_path,
                                    max_epochs=args.max_epochs)
    probability_solver=Probability_Solver(prob_strategy=args.prob_strat,max_epochs=args.M_max_epochs)
    ##准备模型pipeline
    model=prp.model_prepare(activation=activation,
                            layer_width=args.width,
                            repetition_scaler=args.rep_scaler,
                            sample_first=args.sample_first,
                            cpu_gen=cpu_gen,
                            gpu_gen=gpu_gen,
                            device= device,

                            w_b_solver=w_b_solver,
                            adaptive_solver=adaptive_solver,
                            probability_solver=probability_solver,
                            )
    #模型训练
    model.fit(X=X_train,y=u_train)
    #模型保存
    if args.save_weights:
        torch.save(model.state_dict(),os.path.join(experiment_path,"w_b_a.pth"))



    ############################################
    ##############推理阶段#######################
    ############################################

    #模型推理，先用训练原数据推理一次
    SWIM_test.inference(model=model,
                        model_path=experiment_path,
                        X_idn_star=X_idn_star,#X_idn_star
                        u_idn_star=u_idn_star,#u_idn_star
                        T_idn=T_idn,
                        X_idn=X_idn,
                        Exact_idn=Exact_idn,
                        random_seed=args.random_seed,
                        device=device)
    #再用随机生成数据推理5次
    np.random.seed(args.random_seed)
    random_seeds = np.random.choice(np.arange(1, 101), size=5, replace=False).tolist()
    for seed in random_seeds:
        SWIM_test.inference(model_path=experiment_path,device=device,random_seed=seed)

    