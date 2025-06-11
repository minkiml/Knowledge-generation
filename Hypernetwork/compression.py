import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy
import torch

class LayerCompressorBase(object):
    def __init__(self, device = "cpu"):
        super(LayerCompressorBase, self).__init__()
        
        ...
        
    def compress(self, W, name):
        raise NotImplementedError("")
    
    def restoration(self, U, name):
        raise NotImplementedError("")
    
class LayerCompressorSVD(object):
    def __init__(self, k, device = "cpu"):
        '''
        Generally, there are two options in what to predict from hypernet
        
        1. Eigenvalues --> only k dimenional output 
        '''
        super(LayerCompressorSVD, self).__init__()
        self.k = k
        ...
        
        
        self.V = dict()
        
    def compress(self, W):
        dim = W.dim()
        if (dim == 4):
            kernels = W.shape[-1] * W.shape[-2]
        elif (dim == 3):
            kernels = W.shape[-1]
        elif (dim == 2):
            kernels = 1
        else:
            raise ValueError("the input parameter has sigle dimensionality, for which SVD is not applied")
        W = W.view(W.shape[0], W.shape[1], kernels)
        
        Uw = []
        Sw = []
        Vhw = []
        for i in range(kernels):
            U, S, Vh = torch.linalg.svd(W[:,:,i], full_matrices=False)

            Uw.append(U[:,:self.k]) # (H, rank)
            Sw.append(S[:self.k]) # (rank)
            Vhw.append(Vh.T[:self.k]) # (W, rank)
        
        Uw, Sw, Vhw = torch.stack((Uw), dim = -1), torch.stack((Sw), dim = -1), torch.stack((Vhw), dim = -1)
        
        return [Uw.unsqueeze(-1), Vhw.unsqueeze(-1), Sw.unsqueeze(-1)]
    
    def composition(self, SVD_sets):
        raise NotImplementedError("")
    
class LayerCompressorNMF(object):
    def __init__(self, k, device = "cpu"):
        super(LayerCompressorNMF, self).__init__()
        from sklearn.decomposition import NMF
        # nmf = NMF(n_components=k)
        # W = nmf.fit_transform(A.numpy())
        # H = nmf.components_
        # A_approx = W @ H