import torch
print(torch.__version__)              # 应看到 +cu121 后缀
print(torch.cuda.is_available())      # 应输出 True
print(torch.cuda.get_device_name(0))  # 应输出 GeForce RTX 3060 ...