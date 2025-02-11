
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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

        x = self.pool2(F.relu(self.conv2(x) if not self.batchnorm else self.norm_1(self.conv2(x)) )) ### Block 2

        x = F.relu(self.conv3(x) if not self.batchnorm else self.norm_2(self.conv3(x))) ### Block 3

        x = F.relu(self.fc2(self.flat(x))) ### Block 4

        x = self.fc3(x) ### Block 4
        return x

    def initialization_zero(self):
        for param in self.parameters():
            param.data.zero_()