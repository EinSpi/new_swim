import torch
F = torch.tensor([
    [1., 0.],
    [0., 1.],
    [1., 0.]
])
y = torch.tensor([0., 1., 7.])

M = torch.tensor([
    [1., 1., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
])  # 非正交、但方阵

MF = M @ F
My = M @ y

w1 = torch.linalg.lstsq(MF, My).solution
w = torch.linalg.lstsq(F, y).solution

print("w1:", w1)
print("w :", w)