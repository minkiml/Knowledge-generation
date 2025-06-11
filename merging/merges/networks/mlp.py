
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
class MLP_layer(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        """
        MLP
        """
        super(MLP_layer, self).__init__()
        self.linear1 = nn.Linear(in_dim, bottleneck_dim, bias = False)
        self.nl_mlp = nn.LeakyReLU(0.04)
        self.linear2 = nn.Linear(bottleneck_dim, in_dim, bias = False)
    def forward(self, x):
        return self.linear2(self.nl_mlp(self.linear1(x)))
    
class MLP(nn.Module):
    def __init__(self, input_dim = 1, output_dim = 10, 
                 dataset="MNIST", batchnorm = False, zero_init = False):
        super(MLP, self).__init__()
        if dataset == "MNIST":
            image = 28
        elif dataset == "CIFAR-10":
            image = 32
        hidden_dim = 128
        self.in_projection = nn.Linear(input_dim*image**2, hidden_dim, bias = False)
        
        mlp = [MLP_layer(hidden_dim, hidden_dim * 2) for _ in range(2)]
        
        self.layers = nn.ModuleList(mlp)
        layernorm = [nn.LayerNorm(hidden_dim) for _ in range(2)]
        self.layernorms = nn.ModuleList(layernorm)
        self.prediction = nn.Linear(hidden_dim, output_dim, bias = False)
        
    def forward(self, x):
        x = torch.flatten(x, 1, -1)
        
        x = self.in_projection(x)
        for layer, norm in zip(self.layers, self.layernorms):
            x = layer(x)
            x = norm(x)
        y = self.prediction(x)
        return y