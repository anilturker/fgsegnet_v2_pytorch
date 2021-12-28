import torch
import torch.nn as nn
from collections import OrderedDict
import collections
from models.model_tools import SegNetDown, SegNetUp, M_FPM, ConvSig

class FgSegNet(nn.Module):
    """
    Args:
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    def __init__(self, inp_ch, kernel_size=3):
        super().__init__()
        self.model = nn.Sequential()

        # VGG16
        self.enc1 = SegNetDown(inp_ch, 64, 2, batch_norm=False, kernel_size=kernel_size, maxpool=False, dropout=False)
        self.enc2 = SegNetDown(64, 128, 2, batch_norm=False, kernel_size=kernel_size, maxpool=True, dropout=False)
        self.enc3 = SegNetDown(128, 256, 3, batch_norm=False, kernel_size=kernel_size, maxpool=True, dropout=False)
        self.enc4 = SegNetDown(256, 512, 3, batch_norm=False, kernel_size=kernel_size, maxpool=False, dropout=False)

        # FPM module
        self.fpm = M_FPM(512, 64, kernel_size=kernel_size)

        # Decoder
        self.dec3 = SegNetUp(in_ch=320, res_ch=64, out_ch=64, inst_norm=True, kernel_size=kernel_size)
        self.dec2 = SegNetUp(in_ch=64, res_ch=128, out_ch=64, inst_norm=True, kernel_size=kernel_size)
        self.dec1 = SegNetUp(in_ch=64, res_ch=None, out_ch=64, inst_norm=True, kernel_size=kernel_size)
        self.out = ConvSig(64)

        self.frozenLayers = [self.enc1, self.enc2, self.enc3]
        self.apply(self.weight_init)

    def forward(self, inp):
        """
        """
        # Encoder
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # FPM
        e5 = self.fpm(e4)

        # Decoder
        d3 = self.dec3(e5, e1, conv1d=False, upSampling=True)
        d2 = self.dec2(d3, e2, conv1d=True, upSampling=True)
        d1 = self.dec1(d2, conv1d=False, upSampling=False)

        # Classifier
        cd_out = self.out(d1)
        return cd_out



