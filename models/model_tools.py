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
    def __init__(self, in_ch, out_ch, num_rep, batch_norm=False , activation=nn.ReLU(), kernel_size=3,
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
        num_rep (int): Number of repeated conv-inst_norm layers
        inst_norm (bool): Whether to use Instance norm after conv layers
        activation (torch.nn module): Activation function to be used after each conv layer
        kernel_size (int): Size of the convolutional kernels
        dropout (booelan): Whether to apply spatial dropout at the end
    """

    def __init__(self, in_ch, res_ch, out_ch, inst_norm=False, activation=nn.ReLU(),
                 kernel_size=3):

        super().__init__()
        self.up = nn.Sequential()
        self.conv_block = nn.Sequential()
        self.conv1d_block = nn.Sequential()

        if res_ch is not None:
            self.conv1d_block.add_module("conv1d", nn.Conv1d(res_ch, out_ch, kernel_size=(1, 1)))

        self.up.add_module("Upsampling", nn.Upsample(scale_factor=2, mode='nearest'))

        self.conv_block.add_module("conv2d", nn.Conv2d(in_ch, out_ch,
                                                       kernel_size=kernel_size, padding=(int((kernel_size-1)/2))))

        if inst_norm:
            self.conv_block.add_module("inst_norm", nn.InstanceNorm2d(out_ch))

        self.conv_block.add_module("act", activation)


    def forward(self, inp, res=None, conv1d=False, upSampling=False):
        """
        Args:
            inp (tensor): Input tensor
            res (tensor): Residual tensor to be merged, if res=None no skip connections
        """
        feat = self.conv_block(inp)
        if res is None:
            merged =feat
        else:
            if conv1d is True:
                x = self.conv1d_block(res)
            else:
                x = res

            # Global average pooling
            feat_scaled_tensor = []
            avg_feat_tensor = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
            for idx, avg_feat in enumerate(avg_feat_tensor):
                feat_scaled_tensor.append((torch.unsqueeze(feat[idx], 0).permute(0, 2, 3, 1) * avg_feat).permute(0, 3, 1, 2))

            # Adding feature map by scaled one
            feat_scaled_tensor = torch.cat(feat_scaled_tensor, dim=0)
            merged = feat + feat_scaled_tensor

        if upSampling is True:
            output = self.up(merged)
        else:
            output = merged
        return output

class M_FPM(nn.Module):
    """

    """
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()

        in_ch_for_conv = in_ch
        self.fpm_block_1 = nn.Sequential()
        #self.fpm_block_1.add_module("zero pad", nn.ZeroPad2d((0, 1, 0, 1)))
        #self.fpm_block_1.add_module("pool", nn.MaxPool2d(kernel_size=(2,2), stride=(1, 1)))
        self.fpm_block_1.add_module("conv2d", nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=(1, 1)))

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
        self.fpm_out_block.add_module("dropout", nn.Dropout2d(0.25))

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
