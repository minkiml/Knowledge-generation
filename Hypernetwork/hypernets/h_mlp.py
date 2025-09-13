import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

from .hypernet import Hypernet_base

class mul_omega(nn.Module):
    def __init__(self, 
                 feature_dim,
                 omega,
                 omega_learnable = False):
        super(mul_omega, self).__init__() 
        self.omega = nn.Parameter(torch.rand((1,1,feature_dim)) * omega) if omega_learnable else omega

    def forward(self, x):
        return x * self.omega

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    def forward(self, input):
        return torch.sin(input)

class AsymSwiGLU(nn.Module):
     def __init__(self, dim, scale=1.0, mask_num=0):
         super().__init__()
         g = torch.Generator()
         g.manual_seed(abs(hash(str(mask_num)+ str(0))))
         C = torch.randn(dim, dim, generator=g)
         self.register_buffer("C", C)
     def forward(self, x):
         gate = F.sigmoid(F.linear(x, self.C))
         return gate * x

class MLP_layer(nn.Module):
    def __init__(self, in_dim, bottleneck_dim, omega = 1, layer_id = 0):
        """
        MLP
        """
        super(MLP_layer, self).__init__()
        self.linear1 = nn.Linear(in_dim, bottleneck_dim)
        # self.nl_mlp = nn.LeakyReLU(0.05)
        
        self.omega = nn.Identity() 
        # mul_omega(feature_dim = bottleneck_dim,
        #                          omega = omega)
        self.nl_mlp = nn.ReLU() #AsymSwiGLU(bottleneck_dim, mask_num=layer_id) #nn.LeakyReLU(0.02)
        
        self.linear2 = nn.Linear(bottleneck_dim, in_dim)
    def forward(self, x):
        # return self.linear2(self.nl_mlp(self.linear1(x)))
        return self.linear2(self.nl_mlp(self.omega(self.linear1(x))))
    
class Hypernet_MLP(Hypernet_base):
    def __init__(self, param_list, hidden_dim = 128, iteration = 1, num_layer = 1, 
                 node_direction = "W", lowrank = False, rank = 4, type_ = "linear", learnable_emb = False, 
                 zero_init_emb = False, base = "mlp", cond_dim = 0, cond_emb_type = None, masking = False, hyper_grad = False, device = "cpu"):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(Hypernet_MLP, self).__init__(param_list, hidden_dim, iteration, node_direction, lowrank, rank, type_, learnable_emb, 
                                           zero_init_emb, base, cond_dim, cond_emb_type, hyper_grad, device)
        self.cond_dim = cond_dim
        self.num_layer = num_layer
        self.set_transformation()
    def forward_transformation(self, emb):
        
        for _ in range(self.iteration):
            for layer in self.MLP:
                emb = layer(emb)
        return emb
    def set_transformation(self):
        # Transformer network
        MLP = [MLP_layer(self.hidden_dim, self.hidden_dim * 1, layer_id=i) for i in range(self.num_layer)]
        
        self.MLP = nn.ModuleList(MLP)
    