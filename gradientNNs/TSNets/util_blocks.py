import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class baseline_network(nn.Module):
    '''
    Wrapper class
    '''
    def __init__(self, in_channels, num_class, task, revin, channel_dep):
        super(baseline_network, self).__init__()    
        
        
        # Backbone network + linear predictor
        self.network = nn.Linear(in_channels, num_class)
        self.revin = False
        self.channnel_dep = channel_dep
        
        self.softmax = nn.Softmax(dim = 1)
        self.smoothing = 0.25
        self.task = task
        if task=="fault_cls":
            self.criterion=self.criterion_cls
            self.inference = self.inference_cls
            
        elif task=="rul":
            self.criterion=self.criterion_reg
            
        elif task=="fault_det":
            self.criterion=self.criterion_cls
            self.inference = self.inference_cls
            
        elif task == "fault_det_rec":
            self.criterion=self.criterion_rec
            self.inference = self.inference_ad
            self.revin = revin
    def forward(self, x):
        # By default, we expect the input x is in shape of (batch, sequence, channels) 
        if self.task == "fault_det_rec":
            x = self.input_(x)
                
            if self.revin:
                # revin
                x_mean = torch.mean(x, dim=1, keepdim=True)
                x = x - x_mean   
                x_std=torch.sqrt(torch.var(x, dim=1, keepdim=True)+ 1e-5)
                x = x / x_std 
            
        y = self.network(x)
        if self.task == "fault_det_rec":
            if self.revin:
                # reverse revin
                y = y * x_std
                y = y + x_mean 
            y = self.output_(y)
            return y, torch.tensor(0.)
        else:
            return y
    def inference_cls(self, x, feature=False):
        y = self.forward(x)
        y = self.softmax(y)
        y_pred = torch.argmax(y, dim = 1).squeeze(-1)
        return y_pred, None

    def inference_ad(self, x):
        y,_ = self.forward(x)
        # compute anomaly score
        score_TD = ((x - y)**2).mean(dim = -1, keepdim = False).mean(dim = -1, keepdim = False) # (N,)
        return score_TD, torch.tensor(0.)
    
    def criterion_cls(self, logit, target, **null):
        assert target.dim() == 1 and logit.dim() == 2
        return F.cross_entropy(logit, target, label_smoothing=self.smoothing)
    
    def criterion_reg(self, pred, target):
        # MSE
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        return F.mse_loss(pred, target)
    
    def criterion_rec(self, pred, target, **null):
        TD_loss = F.mse_loss(pred, target, reduction = "mean")
        return TD_loss, torch.tensor(0.)
    
        #### Side processings ####
        
    def input_(self, x):
        B, L, C = x.shape
        self.flag_out = False
        if self.channnel_dep == False and C != 1:
            self.flag_out = True
            x = x.permute(0,2,1).reshape(-1, self.C_, L).permute(0,2,1)
            B, L, C = x.shape
        return x

    def output_(self, y):
        B, L_base, C = y.shape
        if self.channnel_dep == False and self.flag_out:
            y = y.permute(0,2,1).reshape(-1, self.C_true, L_base).permute(0,2,1)
        return y
    
class Permute(nn.Module):
    def __init__(self, permutation_order):
        super(Permute, self).__init__()
        self.permutation_order = permutation_order

    def forward(self, x):
        return x.permute(*self.permutation_order)

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()
	def forward(self, x):
		return torch.flatten(x, 1, -1)
        
        
class linear_predictor(nn.Module):
    def __init__(self, in_channels, num_classes, last_channel = False):
        super(linear_predictor, self).__init__()
        # The input to this module expected to be (batch, channel, sequence) by default
        self.last_channel = last_channel
        self.classifier = nn.Sequential(
            Permute((0,2,1)) if self.last_channel else nn.Identity(),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    
class linear_reconstructor(nn.Module):
    def __init__(self, in_channels, out_chnnels, last_channel = True):
        super(linear_reconstructor, self).__init__()
        # The input to this module expected to be (batch, sequence, channel) by default
        self.last_channel = last_channel
        self.classifier = nn.Sequential(
            Permute((0,2,1)) if not self.last_channel else nn.Identity(),
            nn.Linear(in_channels, out_chnnels)
        )

    def forward(self, x):
        return self.classifier(x)
    
