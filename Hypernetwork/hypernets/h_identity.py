import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

from .hypernet import Hypernet_base

class Hypernet_Identity(Hypernet_base):
    def __init__(self, param_list, hidden_dim = 128, iteration = 1, num_layer = 3, 
                 node_direction = "W", lowrank = False, rank = 4, type_ = "linear", learnable_emb = False, 
                 zero_init_emb = False, base = "mlp", cond_dim = 0, cond_emb_type = None, masking = False , hyper_grad = False, device = "cpu"):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(Hypernet_Identity, self).__init__(param_list, hidden_dim, iteration, node_direction, lowrank, rank, type_, learnable_emb, 
                                           zero_init_emb, base, cond_dim, cond_emb_type, hyper_grad, device)
        self.cond_dim = cond_dim
        self.num_layer = num_layer
        self.set_transformation()
    def forward_transformation(self, emb):
        
        return self.id(emb)
    def set_transformation(self):
        # Transformer network        
        self.id = nn.Identity()
    