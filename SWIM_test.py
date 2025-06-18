import argparse
import torch
import utils.pre_processing as prp
import utils.post_processing as psp
import numpy as np
from scipy.interpolate import griddata
import swim_model
import os

def load_config(manifest_path):
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

def inference(model_path:str,device:torch.device,random_seed:int=99,
              model:swim_model.Swim_Model=None,
              X_idn_star:torch.Tensor=None, 
              u_idn_star:torch.Tensor=None, 
              T_idn:torch.Tensor=None, 
              X_idn:torch.Tensor=None, 
              Exact_idn:torch.Tensor=None
              ):
    
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

        _, _, _, _, X_idn_star, u_idn_star, T_idn, X_idn, Exact_idn=prp.load_data(data_path="Data/"+obj+".mat", seed=random_seed)
        
    else:
        pass

    #执行推理
    u_pred_identifier=model(X_idn_star)

    #计算误差并打印
    u_pred_identifier=u_pred_identifier.detach().cpu().numpy()#torch to numpy
    mse=psp.compute_mse(u_pred_identifier,u_idn_star)
    rel_l2=psp.compute_rel_l2(u_pred_identifier,u_idn_star)
    print(f"mse: {mse}")
    print(f"rel l2: {rel_l2}")

    #保存误差至实验目录
    psp.save_errors(save_path=model_path,mse=float(mse),rel_l2=float(rel_l2))

    #绘制推理结果误差图
    lb_idn = np.array([0.0, -20.0])
    ub_idn = np.array([40.0, 20.0])
    keep=1
    U_pred = griddata(X_idn_star, u_pred_identifier.flatten(), (T_idn, X_idn), method='cubic')
    psp.plot_dynamics(ub_idn=ub_idn,lb_idn=lb_idn,Exact_idn=Exact_idn,keep=keep,U_pred=U_pred,save_path=model_path)










if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--random_seed", type=int, default=55)

    args = parser.parse_args()
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference(model_path=args.model_path,random_seed=args.random_seed,device=device)

    
    




