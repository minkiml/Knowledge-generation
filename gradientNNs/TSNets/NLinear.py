import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, revin = True):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin = revin
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = 1
        self.individual = False
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len, bias  = False)
            # nn.Sequential(nn.Linear(self.seq_len, 128, bias  = False),
            #                             nn.LeakyReLU(0.2),
            #                             nn.Linear(128, self.pred_len, bias  = False))#nn.Linear(self.seq_len, self.pred_len, bias  = False)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, L, C = x.shape
        if self.revin:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_std=torch.sqrt(torch.var(x, dim=1, keepdim=True)+ 1e-5)
            x = x / x_std
                
        x = x.permute(0,2,1)
        
        y = self.Linear(x).permute(0,2,1)
        
        if self.revin:
            y = y * x_std
            y = y + x_mean
        return y # [Batch, Output length, Channel]