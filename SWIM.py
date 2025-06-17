import os
import swim_model
import argparse
import activations.activations as act
import utils.pre_processing as prp
import utils.post_processing as psp
import numpy as np
from scipy.interpolate import griddata
import torch
from w_b_solver.w_b_solver import W_B_Solver
from probability_solver.probability_solver import Probability_Solver
from adaptive_solver.adaptive_solver import Adaptive_Solver

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
    parser.add_argument("--experiment", type=str, default="exp616", help="实验名称")
    parser.add_argument("--width", type=int, default=400, help="网络宽度")
    parser.add_argument("--obj", type=str, default="KdV_sine", help="目标函数")
    parser.add_argument("--act",type=str, default="rat",help="激活函数种类")
    parser.add_argument("--p", type=int,default=4,help="有理函数的分子阶数")
    parser.add_argument("--q",type=int,default=3,help="有理函数的分母阶数")
    parser.add_argument("--rep_scaler",type=int,default=2,help="抽样池尺寸是需要神经元数的几倍")
    parser.add_argument("--set_size", type=int,default=7,help="点集大小")
    parser.add_argument("--loss_metric", type=str,default="mse", help="拟合adaptive参数时用的优化目标")
    parser.add_argument("--prob_strat",type=str,default="var",help="用何种标准计算概率")
    parser.add_argument("--reg_factor_1",type=float,default=0.0,help="第一个归一化系数")
    parser.add_argument("--reg_factor_2",type=float,default=1e-6,help="第二个归一化系数")
    parser.add_argument("--int_sketch",type=bool, default=True,help="是否要绘制中间激活函数图像")
    parser.add_argument("--sample_first",type=bool,default=False,help="先抽样还是先拟合")
    parser.add_argument("--optimizer", type=str, default="lbfgs", help="子优化任务优化器")
    parser.add_argument("--random_seed", type=int,default=42,help="随机种子")

    args = parser.parse_args()
    #创建实验目录：
    experiment_path="Results/"+args.experiment+"/"+"obj_"+args.obj+"/"+"act_"+args.act+"/"+"width_"+str(args.width)
    os.makedirs(experiment_path, exist_ok=True)
    #准备训练数据
    X_train, u_train, X_val, u_val, X_idn_star, u_idn_star, T_idn, X_idn, Exact_idn=prp.load_data(data_path="Data/"+args.obj+".mat", seed=42)
    #准备激活函数
    activation=prp.activation_prepare(activation=args.act,p=args.p,q=args.q)
    #准备模型
    ##准备设备
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ##准备随机数生成器
    gpu_gen=torch.Generator(device=device)
    gpu_gen.manual_seed(args.random_seed)
    cpu_gen=torch.Generator()
    cpu_gen.manual_seed(args.random_seed)
    ##准备三个求解器，权重，a_para,概率
    w_b_solver = W_B_Solver(num_interpolation_per_pair=args.set_size)
    adaptive_solver=Adaptive_Solver(loss_metric=args.loss_metric,reg_factor_1=args.reg_factor_1,reg_factor_2=args.reg_factor_2,int_sketch=args.int_sketch,optimizer=args.optimizer,cpu_gen=cpu_gen,save_path=experiment_path)
    probability_solver=Probability_Solver(prob_strategy=args.prob_strat)
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
    #模型推理
    u_pred_identifier=model(X_idn_star)
    #抽查推理结果
    if args.sample_first:
        print("sample first inference result:")
    else:
        print("optimization first inference result:")

    #计算误差并打印
    u_pred_identifier=u_pred_identifier.detach().cpu().numpy()#torch to numpy
    mse=psp.compute_mse(u_pred_identifier,u_idn_star)
    rel_l2=psp.compute_rel_l2(u_pred_identifier,u_idn_star)
    print(f"mse: {mse}")
    print(f"rel l2: {rel_l2}")

    #保存误差至实验目录
    psp.save_errors(save_path=experiment_path,mse=float(mse),rel_l2=float(rel_l2))

    #绘制推理结果误差图
    lb_idn = np.array([0.0, -20.0])
    ub_idn = np.array([40.0, 20.0])
    keep=1
    U_pred = griddata(X_idn_star, u_pred_identifier.flatten(), (T_idn, X_idn), method='cubic')
    psp.plot_dynamics(ub_idn=ub_idn,lb_idn=lb_idn,Exact_idn=Exact_idn,keep=keep,U_pred=U_pred,save_path=experiment_path)












