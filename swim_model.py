from swim_backbones.dense import Dense
from swim_backbones.linear import Linear
from swim_backbones.base import TorchPipeline

class Swim_Model(TorchPipeline):
    def __init__(self, modules):
        super().__init__(modules=modules)
        