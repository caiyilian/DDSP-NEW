import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, norm = "batch"):
        super().__init__()
        if norm == "batch":
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels), 
                nn.LeakyReLU(0.2),
            )
        elif norm == "ins":
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm = "batch"):
        super().__init__()
        if norm == "batch":
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2)
            )
        elif norm == "ins":
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.double_conv(x)

class TransConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm = "batch"):
        super().__init__()
        if norm == "batch":
            self.double_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2),
            )
        elif norm == "ins":
            self.double_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm="ins"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, norm=norm) 
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm="ins"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, norm=norm) 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Resblock3D(nn.Module):
    def __init__(self, in_channels, norm = "batch"):
        super().__init__()
        if norm == "batch":
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.LeakyReLU(0.2),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(in_channels)
            )
        elif norm == "ins":
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(in_channels),
                nn.LeakyReLU(0.2),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(in_channels)
            )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.double_conv(x)+x
        x = self.relu(x)
        return x


def normalization(planes, norm='gn', num_domains=None, momentum=0.1):
    if norm == 'dsbn':
        m = DomainSpecificBatchNorm3d(planes, num_domains=num_domains, momentum=momentum)
    elif norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m


class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_domains)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class DomainSpecificBatchNorm3d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
