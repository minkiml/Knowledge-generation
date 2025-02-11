'''

Differentiable permutation search algorithms (WM-based permutation). 

Adapted from source: Re-basin via implicit Sinkhorn differentiation (https://github.com/fagp/sinkhorn-rebasin/tree/main)
'''
import torch
import torch.nn as nn

class Rebasinnet:
    def __init__(self, 
                 device = None,
                 
                 sequence_model = False):
        super(Rebasinnet, self).__init__()
        pass

        #TODO
        raise NotImplementedError("")

class DistL2Loss(nn.Module):
    '''Cost function for WM'''
    def __init__(self, modela=None):
        super(DistL2Loss, self).__init__()
    #TODO


class DistL1Loss(nn.Module):
    '''Cost function for WM'''
    def __init__(self, modela=None):
        super(DistL1Loss, self).__init__()
    #TODO