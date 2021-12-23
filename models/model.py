
import torch.nn as nn
from models.model_tools import SegNetDown, SegNetUp, M_FPM, ConvSig

class FgSegNet(nn.Module):
    """
    Args:
    """
    def __init__(self, inp_ch, kernel_size=3):
        super().__init__()
        self.model = nn.Sequential()

        # VGG16
        self.enc1 = SegNetDown(inp_ch, 64, 2, batch_norm=False, kernel_size=kernel_size, maxpool=False, dropout=False)
        self.enc2 = SegNetDown(64, 128, 2, batch_norm=False, kernel_size=kernel_size, maxpool=True, dropout=False)
        self.enc3 = SegNetDown(128, 256, 3, batch_norm=False, kernel_size=kernel_size, maxpool=True,)
        self.enc4 = SegNetDown(256, 512, 3, batch_norm=False, kernel_size=kernel_size, maxpool=True, dropout=True)

        self.fpm = M_FPM(512, 64, kernel_size=kernel_size)

        self.dec4 = SegNetUp(320, 256, 64, 2, batch_norm=False, kernel_size=kernel_size)
        self.dec3 = SegNetUp(64, 128, 64, 2, batch_norm=False, kernel_size=kernel_size)
        self.dec2 = SegNetUp(64, 64, 64, 2, batch_norm=False, kernel_size=kernel_size)
        self.dec1 = SegNetUp(64, 64, 64, 2, batch_norm=False, kernel_size=kernel_size)
        self.out = ConvSig(64)


    def forward(self, inp):
        """
        """
        d1 = self.enc1(inp)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)
        d4 = self.enc4(d3)
        d5 = self.fpm(d4)

        u3 = self.dec4(d5, d3)
        u2 = self.dec3(u3, d2)
        u1 = self.dec2(u2, d1)

        cd_out = self.out(u1)
        return cd_out



