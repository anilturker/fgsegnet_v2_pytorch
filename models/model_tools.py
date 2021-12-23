"""
Neural Network Blocks
"""

import torch
import torch.nn as nn

class SegNetDown(nn.Module):
    """ Encoder blocks of SegNet

    Args:
        in_ch (int): Number of input channels for each conv layer
        out_ch (int): Number of output channels for each conv layer
        num_rep (int): Number of repeated conv-batchnorm layers
        batch_norm (bool): Whether to use batch norm after conv layers
        activation (torch.nn module): Activation function to be used after each conv layer
        kernel_size (int): Size of the convolutional kernels
        dropout (booelan): Whether to apply spatial dropout at the end
        maxpool (booelan): Whether to apply max pool in the beginning
    """
    def __init__(self, in_ch, out_ch, num_rep, batch_norm=False, activation=nn.ReLU(), kernel_size=3,
                 dropout=False, maxpool=False):
        super().__init__()
        self.down_block = nn.Sequential()

        if maxpool:
            self.down_block.add_module("maxpool", nn.MaxPool2d(2))
        in_ch_for_conv = in_ch
        for k in range(num_rep):
            self.down_block.add_module("conv%d"%(k+1), nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, padding=(int((kernel_size-1)/2))))
            self.down_block.add_module("act%d"%(k+1), activation)
            if batch_norm:
                self.down_block.add_module("bn%d"%(k+1), nn.BatchNorm2d(out_ch))
            in_ch_for_conv = out_ch
        if dropout:
            self.down_block.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, inp):
        return self.down_block(inp)

class SegNetUp(nn.Module):
    """ Decoder blocks of UNet

    Args:
        in_ch (int): Number of input channels for each conv layer
        res_ch (int): Number of channels coming from the residual, if equal to 0 and no skip connections
        out_ch (int): Number of output channels for each conv layer
        num_rep (int): Number of repeated conv-batchnorm layers
        batch_norm (bool): Whether to use batch norm after conv layers
        activation (torch.nn module): Activation function to be used after each conv layer
        kernel_size (int): Size of the convolutional kernels
        dropout (booelan): Whether to apply spatial dropout at the end
    """

    def __init__(self, in_ch, res_ch, out_ch, num_rep, batch_norm=False, activation=nn.ReLU(), kernel_size=3,
                 dropout=False):

        super().__init__()
        self.up = nn.Sequential()
        self.conv_block = nn.Sequential()

        self.up.add_module("conv2d_transpose", nn.ConvTranspose2d(in_ch, in_ch, kernel_size, stride=2,
                                                                  output_padding=(int((kernel_size-1)/2)),
                                                                  padding=(int((kernel_size-1)/2))))
        if batch_norm:
            self.up.add_module("bn1", nn.BatchNorm2d(in_ch))

        in_ch_for_conv = in_ch + res_ch
        for k in range(num_rep):
            self.conv_block.add_module("conv%d"%(k+1), nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, padding=(int((kernel_size-1)/2))))
            self.conv_block.add_module("act%d"%(k+1), activation)
            if batch_norm:
                self.conv_block.add_module("bn%d"%(k+2), nn.BatchNorm2d(out_ch))
            in_ch_for_conv = out_ch
        if dropout:
            self.conv_block.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, inp, res=None):
        """
        Args:
            inp (tensor): Input tensor
            res (tensor): Residual tensor to be merged, if res=None no skip connections
        """
        feat = self.up(inp)
        if res is None:
            merged = feat
        else:
            merged = torch.cat([feat, res], dim=1)
        return self.conv_block(merged)

class M_FPM(nn.Module):
    """

    """
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()

        in_ch_for_conv = in_ch
        self.fpm_block_1 = nn.Sequential()
        self.fpm_block_1.add_module("conv2d", nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, padding=(int((kernel_size - 1)/2))))

        self.fpm_block_2 = nn.Sequential()
        self.fpm_block_2.add_module("conv2d", nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, padding=(int((kernel_size - 1)/2))))

        in_ch_for_conv = in_ch + out_ch
        dilation = 4
        self.fpm_block_3 = nn.Sequential()
        self.fpm_block_3.add_module("act", nn.ReLU())
        self.fpm_block_3.add_module("conv2d", nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, dilation=dilation,
                                                        padding=dilation))
        in_ch_for_conv = in_ch + out_ch
        dilation = 8
        self.fpm_block_4 = nn.Sequential()
        self.fpm_block_4.add_module("act", nn.ReLU())
        self.fpm_block_4.add_module("conv2d", nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, dilation=dilation,
                                                        padding=dilation))
        in_ch_for_conv = in_ch + out_ch
        dilation = 16
        self.fpm_block_5 = nn.Sequential()
        self.fpm_block_5.add_module("act", nn.ReLU())
        self.fpm_block_5.add_module("conv2d", nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, dilation=dilation,
                                                        padding=dilation))

        self.fpm_out_block = nn.Sequential()
        self.fpm_out_block.add_module("inst_norm", nn.InstanceNorm2d(in_ch))
        self.fpm_out_block.add_module("act", nn.ReLU())

    def forward(self, inp):
        res_1 = self.fpm_block_1(inp)
        res_2 = self.fpm_block_2(inp)

        merged = torch.cat([inp, res_2], dim=1)
        res_3 = self.fpm_block_3(merged)

        merged = torch.cat([inp, res_3], dim=1)
        res_4 = self.fpm_block_4(merged)

        merged = torch.cat([inp, res_4], dim=1)
        res_5 = self.fpm_block_5(merged)

        merged = torch.cat([res_1, res_2, res_3, res_4, res_5], dim=1)
        fpm_out = self.fpm_out_block(merged)

        return fpm_out

class ConvSig(nn.Module):
    """ Conv layer + Sigmoid

    Args:
        in_ch (int): Number of input channels
    """

    def __init__(self, in_ch):
        super().__init__()
        self.out = nn.Sequential()
        self.out.add_module("conv2d", nn.Conv2d(in_ch, 1, 1))
        self.out.add_module("sigmoid", nn.Sigmoid())

    def forward(self, inp):
        return self.out(inp)
