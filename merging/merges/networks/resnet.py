import torch.nn as nn
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
def conv3x3(in_channels, out_channels, stride=1, padding=1)-> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                        stride=stride, padding=padding, bias=False)

# Residual block for ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        if downsample is None:
            self.downsample = nn.Identity() 
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample:
        residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        return out

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Downsample, self).__init__()
        self.conv_down = conv3x3(in_channels, out_channels, stride=stride)
        self.bn_down = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.bn_down(x)
        return x

class Input_layer(nn.Module):
    def __init__(self, input_dim, in_channels):
        super(Input_layer, self).__init__()
        self.conv = conv3x3(input_dim, in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class Output_layer(nn.Module):
    def __init__(self, num_classes, in_channel = 64):
        super(Output_layer, self).__init__()
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(in_channel, num_classes)
    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# ResNet 18 for cifar10 similar to the one the main paper tells us , 
# equal to the one presented in https://arxiv.org/pdf/1512.03385.pdf.
class ResNet18_cifar(nn.Module):
    def __init__(self, input_dim = 3, num_classes=10, zero_init = False, multiplier = 1
                    ):
        super(ResNet18_cifar, self).__init__()
        block = ResidualBlock
        layers = [3, 3, 3]
        multiplier = multiplier
        default_in_channel = 16
        self.in_channels = int(default_in_channel * multiplier)
        self.input_layer = Input_layer(input_dim, self.in_channels)
        self.layer1 = self.make_layer(block, int(default_in_channel * multiplier), layers[0])
        self.layer2 = self.make_layer(block, int(32 * multiplier), layers[1], 2)
        self.layer3 = self.make_layer(block, int(64 * multiplier), layers[2], 2)
        self.output_layer = Output_layer(num_classes, in_channel = int(64 * multiplier))

        if zero_init:
            self.initialization_zero()
        else:
            self.apply(_weights_init)
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = Downsample(self.in_channels, out_channels, stride=stride)
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return out
    
    def initialization_zero(self):
        for param in self.parameters():
            param.data.zero_()
