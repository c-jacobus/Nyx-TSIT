import torch
import torch.nn as nn
from torch.nn import functional as F


class PeriodicPad3d(nn.Module):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad3d, self).__init__()
       self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0, 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width, 0, 0), mode="constant", value=0)
        # pad front and back zeros
        out = F.pad(out, (0, 0, 0, 0, self.pad_width, self.pad_width), mode="constant", value=0)
        return out

