import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .hypernet import Hypernet_base

class Hypernet_LSTM(Hypernet_base):
    def __init__(self, param_list, hidden_dim = 128, iteration = 1,
                 num_layer = 1, node_direction = "W", lowrank = False, rank = 4, type_ = "linear", learnable_emb = False,
                 zero_init_emb = False,
                 device = "cpu"):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(Hypernet_LSTM, self).__init__(param_list, hidden_dim, iteration, node_direction, lowrank, rank, type_, learnable_emb,
                                            zero_init_emb,
                                           device)
        self.num_layer = num_layer
        self.set_transformation()
        
    def forward_transformation(self, emb):
        for _ in range(self.iteration):
            emb, _ = self.LSTM(emb)
        return emb
    
    def set_transformation(self):
        # Transformer network
        
        self.LSTM = nn.LSTM(
        input_size=self.hidden_dim,   # dz
        hidden_size=self.hidden_dim,  # same as d_model
        num_layers=self.num_layer,    # like your Transformer layer count
        batch_first=True,
        dropout=0.0,
        device=self.device
        )
        