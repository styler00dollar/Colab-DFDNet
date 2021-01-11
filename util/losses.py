from collections import namedtuple
import torch


LossOutputVgg19 = namedtuple("LossOutput", ["relu_6", "relu_11", "relu_20", "relu_29"])

class LossNetworkVgg19(torch.nn.Module):
    def __init__(self, vgg19_model):
        super(LossNetworkVgg19, self).__init__()
        self.vgg_layers = vgg19_model.features
        self.layer_name_mapping = {
            '6': "relu_6",
            '11': "relu_11",
            '20': "relu_20",
            '29': "relu_29"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutputVgg19(**output)