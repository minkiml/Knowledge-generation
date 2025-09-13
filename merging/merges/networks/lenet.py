import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, learnable_affine = True):
        super().__init__()
        self.eps = eps
        self.learnable_affine = learnable_affine
        if learnable_affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # x shape (B, C, H, W)
        # Mean and var over channel dimension only
        mean = x.mean(dim=1, keepdim=True)  # shape: (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.learnable_affine:
            return self.gamma * x_norm + self.beta
        else:
            return x_norm
        
class reshape_(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        # x shape (B, C, H, W) -> (B, *, C)
        return x.permute(0,2,3,1)
class reverse_reshape_(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,x):
        return x.permute(0,3,1,2)
# Test net
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
class LeNet(nn.Module):
    def __init__(self, input_dim = 1, output_dim = 10, 
                 dataset="MNIST", batchnorm = False, zero_init = False):
        super(LeNet, self).__init__()
        # 6 kernels 5x5
        self.conv1 = nn.Conv2d(
            input_dim,
            6,
            5,
            padding="valid",
        )
        self.batchnorm = batchnorm
        # max-pooling over 2x2
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # 16 kernels 5x5
        self.conv2 = nn.Conv2d(6, 16, 5, padding="valid")
        if batchnorm:
            self.norm_1 = nn.BatchNorm2d(16)
        else:
            self.norm_1 = nn.Sequential(reshape_(),
                                        nn.LayerNorm(16),
                                        reverse_reshape_()) #ChannelwiseLayerNorm(16, learnable_affine=True)
        # max-pooling over 2x2
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # 120 kernels 4x4 to match the dimensionality of the fully connected network
        self.conv3 = nn.Conv2d(
            16,
            120,
            5 if dataset == "CIFAR-10" else 4,
        )
        if batchnorm:
            self.norm_2 = nn.BatchNorm2d(120)
        else:
            self.norm_2 = nn.Sequential(reshape_(),
                                        nn.LayerNorm(120),
                                        reverse_reshape_()) #ChannelwiseLayerNorm(120, learnable_affine=True)
        # 120 fully connected neurons, too many parameter in this case w.r.t. the paper
        # self.fc1 = nn.Linear(16 * 5 * 5, 120,)
        self.flat = nn.Flatten(start_dim=1)
        # 84 fully connected neurons
        self.fc2 = nn.Linear(120, 84)
        # 10 fully connected neurons
        self.fc3 = nn.Linear(
            84,
            output_dim,
        )

        if zero_init:
            self.initialization_zero()
        else:
            self.apply(_weights_init)
    def forward(self, x):
        # x = x.view(-1, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x))) ### Block 1

        x = self.pool2(F.relu(self.norm_1(self.conv2(x)) )) ### Block 2

        x = F.relu(self.norm_2(self.conv3(x))) ### Block 3

        x = F.relu(self.fc2(self.flat(x))) ### Block 4

        x = self.fc3(x) ### Block 4
        return x

    def initialization_zero(self):
        for param in self.parameters():
            param.data.zero_()