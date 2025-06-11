import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .hypernet import Hypernet_base
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
     
class Hypernet_mixer(Hypernet_base):
    def __init__(self, param_list, hidden_dim = 128, iteration = 1,
                 num_layer = 1, node_direction = "W", lowrank = False, rank = 4, type_ = "linear", learnable_emb = False,
                 device = "cpu", mainnet_depth = 1, zero_init_emb = False):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(Hypernet_mixer, self).__init__(param_list, hidden_dim, iteration, node_direction, lowrank, rank, type_, learnable_emb,
                                             zero_init_emb,
                                           device)
        self.num_layer = num_layer
        self.set_transformation()
        
    def forward_transformation(self, emb):
        for _ in range(self.iteration):
            emb = self.MLPMIXER(emb)
        return emb
    
    def set_transformation(self):
        # Transformer network
        self.MLPMIXER = MLPMixer(d_model = self.hidden_dim, 
                        num_layers = self.num_layer,
                        depth = self.depth)
    def generate_mask(self, L):
        # Example: upper-triangular mask (causal attention)
        # For encoder you might use padding mask or a custom mask
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
        return mask.to(self.device)

mlpmixer_args = {
        "in_channels":3,
        "img_size":32, 
        "mainent_depth":4, 
        "hidden_size":128, 
        "hidden_s":64, 
        "hidden_c":384, 
        "num_layers":3, 
        "num_classes":10, 
        "drop_p":0.,
        "off_act":False,
        "is_cls_token":False
        }
class MLPMixer(nn.Module):
    def __init__(self, d_model, num_layers, depth):
        super(MLPMixer, self).__init__()

        # Arguments
        mainent_depth=depth
        hidden_size=d_model
        hidden_s= 64 # mlpmixer_args["hidden_s"]
        hidden_c= d_model * 3
        num_layers=num_layers
        drop_p=mlpmixer_args["drop_p"]
        off_act=mlpmixer_args["off_act"]

        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(mainent_depth, hidden_size, hidden_s, hidden_c, drop_p, off_act) 
            for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        out = self.mixer_layers(x)
        out = self.ln(out)
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

# Spacial
class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

# chnnel
class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x
