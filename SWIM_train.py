import os
import argparse
import utils.pre_processing as prp
import utils.post_processing as psp
import numpy as np
import torch
from solvers import *
import SWIM_infer
import time

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
    parser.add_argument("--init_method",type=str,default="relu_like",help="怎样初始化a_params,如果有")
    parser.add_argument("--optimizer", type=str, default="adam", help="子优化任务优化器")
    parser.add_argument("--p", type=int,default=4,help="有理函数的分子阶数")
    parser.add_argument("--q",type=int,default=3,help="有理函数的分母阶数")
    parser.add_argument("--set_size", type=int,default=7,help="点集大小")

    parser.add_argument("--max_epoch",type=int,default=3000,help="局部优化adaptive parameter最大迭代次数")
    parser.add_argument("--M_max_epoch",type=int,default=500,help="优化M矩阵最大迭代次数")
    parser.add_argument("--reg_factor",type=float,default=1e-6,help="第二个归一化系数")
    
    parser.add_argument("--random_seed", type=int,default=92,help="随机种子")
    parser.add_argument("--int_sketch",type=bool, default=True,help="是否要绘制中间激活函数图像")
    parser.add_argument("--save_weight", type=bool,default=True, help="是否要保存训练后的权重")
    parser.add_argument("--device",type=int, default=0, help="实验要使用的设备编号") 

    parser.add_argument("--path_keys", type=str, nargs='+', default=["exp","obj","act","width"],help="哪些参数参与实验统计")
    
    args = parser.parse_args()

    #########################################
    ############目录准备阶段##################
    ########################################
    #创建实验目录：
    #所有参数列表
    all_arg_names=["exp","obj","act","width",
                   "rep_scaler","loss_metric","prob_strat","init_method","optimizer",
                   "p","q","set_size","max_epoch",
                   "M_max_epoch","reg_factor","random_seed",
                   "int_sketch","save_weight","device"]
    experiment_path =os.path.join("Results",args.exp)
    for name in all_arg_names:
        if name in args.path_keys:
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
                                    cpu_gen=cpu_gen,gpu_gen=gpu_gen,
                                    init_method=args.init_method,
                                    save_path=experiment_path,
                                    max_epochs=args.max_epoch)
    probability_solver=Probability_Solver(prob_strategy=args.prob_strat,max_epochs=args.M_max_epoch)
    ##准备模型pipeline
    model=prp.model_prepare(activation=activation,
                            layer_width=args.width,
                            repetition_scaler=args.rep_scaler,
                            cpu_gen=cpu_gen,
                            gpu_gen=gpu_gen,
                            device= device,

                            w_b_solver=w_b_solver,
                            adaptive_solver=adaptive_solver,
                            probability_solver=probability_solver,
                            )
    #模型训练
    start_time = time.time()#计时
    model.fit(X=X_train,y=u_train)
    end_time = time.time()
    dauer=end_time-start_time
    psp.save_training_time(save_path=experiment_path,dauer=dauer)
    
              
    #模型保存
    if args.save_weight:
        torch.save(model.state_dict(),os.path.join(experiment_path,"w_b_a.pth"))



    ############################################
    ##############推理阶段#######################
    ############################################

    #模型推理，先用训练原数据推理一次
    SWIM_infer.inference(model=model,
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
        SWIM_infer.inference(model_path=experiment_path,device=device,random_seed=seed)

    