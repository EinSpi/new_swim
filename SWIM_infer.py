import argparse
import torch
import utils.pre_processing as prp
import utils.post_processing as psp
import numpy as np
from scipy.interpolate import griddata
from .swim_backbones import Swim_Model
import os

def load_config(manifest_path):
    #从manifest文件获取模型信息，加载模型
    config = {}
    with open(os.path.join(manifest_path,"manifest.txt"), "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            if key in {"act", "p", "q","obj","width"}:
                config[key] = value
    # 转换数值类型
    config["p"] = int(config["p"])
    config["q"] = int(config["q"])
    config["width"] = int(config["width"])
    
    return config  # dict like {'act': 'rat', 'p': 4, 'q': 3}
def custom_random_data(lb=np.array([0.0, -20.0]),ub=np.array([40.0, 20.0]),random_seed:int=42,N_total:int=20000)->torch.Tensor:
    """
    通过下界/上界生成随机 (t, x) 采样点
    """
    np.random.seed(random_seed)
    # 取出范围
    t_range = [lb[0], ub[0]]
    x_range = [lb[1], ub[1]]

    # 随机采样点
    t_idn_star = np.random.uniform(t_range[0], t_range[1], size=(N_total, 1))
    x_idn_star = np.random.uniform(x_range[0], x_range[1], size=(N_total, 1))
    X_idn_star = np.hstack((t_idn_star, x_idn_star))

    X_idn_star = torch.tensor(X_idn_star,dtype=torch.float32)

    return X_idn_star

def inference(model_path:str,device:torch.device,random_seed:int=99,
              model:Swim_Model=None,
              X_idn_star:torch.Tensor=None, 
              u_idn_star:torch.Tensor=None, 
              T_idn:torch.Tensor=None, 
              X_idn:torch.Tensor=None, 
              Exact_idn:torch.Tensor=None
              ):
    lb_idn = np.array([0.0, -20.0])
    ub_idn = np.array([40.0, 20.0])
    keep=1
    
    if model==None or X_idn_star==None:
        config = load_config(model_path)
        act = config["act"]
        p = config["p"]
        q = config["q"]
        obj =config["obj"]
        layer_width=config["width"]

        #根据已加载信息组建模型
        ##准备激活函数
        activation=prp.activation_prepare(activation=act,p=p,q=q)
        ##创建模型
        model=prp.model_prepare(activation=activation,device=device,layer_width=layer_width)
        ##加载权重，包括w,b,a_paras
        model.load_state_dict(torch.load(os.path.join(model_path,"w_b_a.pth"),map_location=device))
        model = model.to(device)
        model.eval()

        _, _, _,_, _, _, _, T_idn, X_idn, Exact_idn=prp.load_data(data_path="Data/"+obj+".mat", seed=random_seed)
        X_idn_star=custom_random_data(lb=lb_idn,ub=ub_idn ,random_seed=random_seed,N_total=100000)
        u_pred_identifier=model(X_idn_star)#(100000,1)
        u_pred_identifier=u_pred_identifier.detach().cpu().numpy()#torch to numpy #(100000,1)
        U_pred = griddata(X_idn_star, u_pred_identifier.flatten(), (T_idn, X_idn), method='nearest')#插值到格点上方便后续比对

        mse = psp.compute_mse(U_pred,Exact_idn)
        rel_l2  = psp.compute_rel_l2(U_pred,Exact_idn)
    else:
        #直接执行推理
        u_pred_identifier=model(X_idn_star)
        #计算误差并打印
        u_pred_identifier=u_pred_identifier.detach().cpu().numpy()#torch to numpy
        mse=psp.compute_mse(u_pred_identifier,u_idn_star)
        rel_l2=psp.compute_rel_l2(u_pred_identifier,u_idn_star)
        #绘制推理结果误差图
        U_pred = griddata(X_idn_star, u_pred_identifier.flatten(), (T_idn, X_idn), method='nearest')
        

    print("model "+model_path+f" mse: {mse} rel l2: {rel_l2}")
    psp.save_errors(save_path=model_path,mse=float(mse),rel_l2=float(rel_l2))
    psp.plot_dynamics(ub_idn=ub_idn,lb_idn=lb_idn,Exact_idn=Exact_idn,keep=keep,U_pred=U_pred,save_path=model_path)









#从命令行启动
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--random_seed", type=int, default=55)

    args = parser.parse_args()
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference(model_path=args.model_path,random_seed=args.random_seed,device=device)

    
    




