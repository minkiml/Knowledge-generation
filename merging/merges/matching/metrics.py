
'''
Computing Cost metrics for non-differentiable permutations
'''
import torch 
import pdb

def FeatureReshapeHandler(x, class_):
    """ To reshape layer intermediates for alignment metric computation. """

    def handle_conv2d(x):
        # reshapes conv2d representation from [B, C, H, W] to [C, -1]
        B, C, H, W = x.shape
        return x.permute(1, 0, 2, 3).reshape(C, -1)

    def handle_linear(x):
        # x is shape [..., C]. Want [C, -1]
        x = x.flatten(0, len(x.shape)-2).transpose(1, 0).contiguous()
        return x
    # TODO: using class as argument could be ambiguous. consider using just the input shape. 
    # the arguments are parametric module class
    reshaper = {"Linear": handle_linear,
                "Conv2d": handle_conv2d,
                "LayerNorm": handle_linear,
                "BatchNorm2d": handle_conv2d,
                "Conv1D": handle_linear
                }[class_]
    return reshaper(x = x)

class Pairwise_covariance(object):
    def __init__(self, correlation = True):
        self.correlation = correlation

        self.std = None
        self.mean = None
        self.outer = None
        self.similarity = None

    def update(self, *feats, class_):
        # Reshape intermediate features to (channel, *). E.g., (1, 3, 28, 28) --> (3, 1 * 28 * 28)
        feats = feats[0]
        sample_size = feats[0].shape[0]
        for i, f in enumerate(feats):
            
            f = FeatureReshapeHandler(f, class_)
            feats[i] = f

        feats = torch.cat((feats), dim=0)
        feats = torch.nan_to_num(feats, 0, 0, 0)

        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]
        
        if self.mean  is None: self.mean  = torch.zeros_like( mean)
        if self.outer is None: self.outer = torch.zeros_like(outer)
        if self.std   is None: self.std   = torch.zeros_like(  std)

        # Multiplying by sample_size to make sure the updates are correctly scaled
        self.mean  += mean  * sample_size
        self.outer += outer * sample_size
        self.std   += std   * sample_size


    def finalize(self, numel, eps=1e-4):
        self.outer /= numel
        self.mean  /= numel
        self.std   /= numel
        cov = self.outer - torch.outer(self.mean, self.mean)

        if torch.isnan(cov).any():
            print('NaN found in covariance metric finalize.', flush=True)
            breakpoint()
        if (torch.diagonal(cov) < 0).sum():
            print(f'Torch diag size {len(torch.diagonal(cov))} and sum of under 0 values is {(torch.diagonal(cov) < 0).sum()}', flush=True)

        def compute_correlation(covariance, eps=1e-7):
            std = torch.diagonal(covariance).sqrt()
            covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
            return covariance
        
        if self.correlation:
            cov = compute_correlation(cov.squeeze())
        self.similarity = cov
        return cov # (2feature by 2feature)
    

    def get(self):
        return self.similarity
    