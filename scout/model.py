"""
Scout model template.
"""

import torch.nn as nn
from torch import sigmoid
from torch.jit import script
from torch.autograd import Function
from torch.nn import ReLU
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout


@script
def swish_jit_fwd(x):
    return x * sigmoid(x)


@script
def swish_jit_bwd(x, grad):
    x_s = sigmoid(x)
    return grad * (x_s * (1 + x * (1 - x_s)))


class SwishAutoFn(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad)


class Swish(Module):
    """
    Swish Activation function

    https://arxiv.org/abs/1710.05941
    """
    def forward(self, x):
        return SwishAutoFn.apply(x)


activations = {
    "relu": ReLU,
    "swish": Swish,
}


class Model(Module):
    """
    Model template for QuartzNet style architectures

    https://arxiv.org/pdf/1910.10261.pdf
    """
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        features = config['input']['features']
        activation = activations[ config['network']['activation'] ]()

        layers = []
        for layer in self.config['block']:
            layers.append(
                Block(
                    features, layer['filters'], activation,
                    repeat=layer['repeat'], kernel_size=layer['kernel'],
                    stride=layer['stride'], dilation=layer['dilation'],
                    dropout=layer['dropout'], residual=layer['residual'],
                    separable=layer['separable'], padding=layer['padding'],
                )
            )
            features = layer['filters']
        layers.append(Conv1d(features, 1, kernel_size=1, bias=True))

        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class TCSConv1d(Module):
    """
    Time-Channel Separable 1D Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
            padding=0, dilation=1, groups=1, bias=False, separable=False):

        super(TCSConv1d, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = Conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias, groups=in_channels
            )

            self.pointwise = Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride,
                dilation=dilation, bias=bias, padding=0
            )
        else:
            self.conv = Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x



class Block(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, 
            activation, repeat=5, kernel_size=1, stride=1, dilation=1, 
            dropout=0.0, residual=False, separable=False, padding=0):

        super(Block, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()

        _in_channels = in_channels
        if self.use_res: # override padding so residual layer works
            padding = self.get_padding(kernel_size[0], stride[0], dilation[0])

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable
                )
            )

            self.conv.extend(self.get_activation(activation, dropout))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable
            )
        )

        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs(in_channels, out_channels))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation(activation, dropout))


    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)


    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation


    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, 
            dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]


    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x += self.residual(x)
        return self.activation(_x)