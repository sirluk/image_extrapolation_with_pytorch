import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Union    
    

class ConvAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

        
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]
               
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.channels, self.activation = channels, activation

        self.activate = activation_func(activation)
        self.transform = nn.Sequential(
            ConvAuto(in_channels=channels, out_channels=channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(channels),
            self.activate,
            ConvAuto(in_channels=channels, out_channels=channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        residual = x
        x = self.transform(x)
        x += residual
        x = self.activate(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, activation="relu"):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_func(activation)
        )
        
    def forward(self, x):
        return self.conv(x)
    

class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation="relu"):
        super().__init__()
        
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_func(activation)
        )
        
    def forward(self, x):
        return self.transpose_conv(x)

        
class MultiConv(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 32, kernel_sizes: tuple = (7,), bias: bool = True):    
        super(MultiConv, self).__init__()
                
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.bias = bias
        
        convs = []
        for kernel_size in kernel_sizes:
            convs.append(ConvAuto(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias))
        self.convs = nn.Sequential(*convs)
            
        #1x1 convolution to map concatenated output of multiconv back to shape n_kernels
        self.conv_1x1 = nn.Conv2d(in_channels=(out_channels * len(kernel_sizes)), out_channels=out_channels, kernel_size=1, bias=True)
            
    def forward(self, x):
        x_ = torch.cat([c(x) for c in self.convs], dim=1)
        return self.conv_1x1(x_)
    
    
class CNNBase(nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: Union[tuple,int] = 7, kernel_size_out: int = 7, batch_norm: bool = False, activation="relu"):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(CNNBase, self).__init__()
        
        if type(kernel_size)==tuple:
            if len(kernel_size)!=n_hidden_layers:
                raise Exception("when defining different kernel sizes, number of kernels must be equal to number of hidden layers")
        elif type(kernel_size)==int:
            kernel_size = [kernel_size] * n_hidden_layers
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(ConvAuto(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size[i], bias=not batch_norm))
            if batch_norm: cnn.append(nn.BatchNorm2d(n_kernels))
            cnn.append(activation_func(activation))
            n_in_channels = n_kernels
        self.hidden_layers = nn.Sequential(*cnn)
        self.output_layer = ConvAuto(in_channels=n_in_channels, out_channels=1, kernel_size=kernel_size_out, bias=True)
    
    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred
        

class CNNBaseMulti(nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_sizes: Union[tuple, int] = (7,), batch_norm: bool = False, kernel_size_out: int = 7, activation="relu"):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(CNNBaseMulti, self).__init__()
        
        if type(kernel_sizes)==int:
            kernel_sizes = tuple([kernel_sizes])
        elif type(kernel_sizes)==list:
            kernel_sizes = tuple(kernel_sizes)
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(MultiConv(in_channels=n_in_channels, out_channels=n_kernels, kernel_sizes=kernel_sizes, bias=not batch_norm))
            if batch_norm: cnn.append(nn.BatchNorm2d(n_kernels))
            cnn.append(activation_func(activation))
            n_in_channels = n_kernels
        self.hidden_layers = nn.Sequential(*cnn)

        self.output_layer = ConvAuto(in_channels=n_in_channels, out_channels=1, kernel_size=kernel_size_out, bias=True)

    
    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred
    
    
class BorderPredictionNet(nn.Module):
    def __init__(self, cnn: nn.Module):
        super(BorderPredictionNet, self).__init__()
        
        self.cnn = cnn
        
    def forward(self, x, mask):
        x_ = self.cnn(x).squeeze(1)
        return pad_sequence([o[~m] for o, m in zip(x_, mask)], batch_first=True)

    
