import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.architecture import StreamResnetBlock


# Content/style stream.
# The two streams are symmetrical with the same network structure,
# aiming at extracting corresponding feature representations in different levels.
class Stream(BaseNetwork):
    def __init__(self, params, reshape_size=None):
        super().__init__()
        self.params = params
        self.ppad = self.params.use_periodic_padding
        self.StreamResnetBlock = StreamResnetBlock
        nf = params.ngf
        
        self.res_0 = self.StreamResnetBlock(params.input_nc, 1 * nf, params)  # 64-ch feature
        self.res_1 = self.StreamResnetBlock(1  * nf, 2  * nf, params)   # 128-ch  feature
        self.res_2 = self.StreamResnetBlock(2  * nf, 4  * nf, params)   # 256-ch  feature
        self.res_3 = self.StreamResnetBlock(4  * nf, 8  * nf, params)   # 512-ch  feature
        self.res_4 = self.StreamResnetBlock(8  * nf, 16 * nf, params)   # 1024-ch feature
        self.res_5 = self.StreamResnetBlock(16 * nf, 16 * nf, params)   # 1024-ch feature
        self.res_6 = self.StreamResnetBlock(16 * nf, 16 * nf, params)   # 1024-ch feature
        self.res_7 = self.StreamResnetBlock(16 * nf, 16 * nf, params) if params.num_upsampling_blocks != 6 else None   # 1024-ch feature

    def down(self, input):
        return F.interpolate(input, scale_factor=0.5)

    def forward(self,input):
        describe = False
        if describe: print('Stream Input: {}'.format(input.size()))
        # assume that input shape is (n,c,256,512)
        
        x0 = self.res_0(input) # (n,64,128^3)
        if describe: print('stream 0: {}'.format(x0.size()))
        
        x1 = self.down(x0)
        x1 = self.res_1(x1)    # (n,128,64^3)
        if describe: print('stream 1: {}'.format(x1.size()))
        
        x2 = self.down(x1)
        x2 = self.res_2(x2)    # (n,256,32^3)
        if describe: print('stream 2: {}'.format(x2.size()))

        x3 = self.down(x2)
        x3 = self.res_3(x3)    # (n,512,16^3)
        if describe: print('stream 3: {}'.format(x3.size()))

        x4 = self.down(x3)
        x4 = self.res_4(x4)    # (n,1024,8^3)
        if describe: print('stream 4: {}'.format(x4.size()))

        x5 = self.down(x4)
        x5 = self.res_5(x5)    # (n,1024,4^3)
        if describe: print('stream 5: {}'.format(x5.size()))
        
        if self.params.num_upsampling_blocks >= 6:
            x6 = self.down(x5)
            x6 = self.res_6(x6)    # (n,1024,2^)
            if describe: print('stream 6: {}'.format(x6.size()))
        else:
            x6 = None

        if self.params.num_upsampling_blocks >= 7:
            x7 = self.down(x6)
            x7 = self.res_7(x7)    # (n,1024,2,4)
            if describe: print('stream 7: {}'.format(x7.size()))
        else:
            x7 = None
        
        return [x0, x1, x2, x3, x4, x5, x6, x7]


# Additive noise stream inspired by StyleGAN.
# The two streams are symmetrical with the same network structure,
# aiming at extracting corresponding feature representations in different levels.

class NoiseStream(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        nf = params.ngf

        #iloc, isc = 1., 0.05
        iloc = params.iloc
        isc = params.isc
        
        scalers = []
        for i in range(8):
            val = torch.from_numpy(np.random.normal(loc=np.sqrt(iloc), scale=isc, size=(1,nf,1,1,1)).astype(np.float32))
            nn.Parameter(val, requires_grad=params.learnable_noise)
            scalers.append(nn.Parameter(val, requires_grad=True))
            nf = min(nf*2, self.params.ngf*16)
            
        self.featmult = nn.ParameterList(scalers)

    def forward(self,input):
        # assume that input shape is (n,c,h,w,d)
        n,h,w,d = input.shape[0], input.shape[2], input.shape[3], input.shape[4]

        nf = self.params.ngf
        out = []
        for i in range(8):
            noise = torch.randn((n, 1, h, w, d), device=input.device)
            
            #print('epoch: {}'.format(self.epoch))
            
            sc =  float(self.params.noise_scale)#+(self.epoch*self.params.Noise_schedule)
            out.append(noise*sc*self.featmult[i])
            
            #out.append(noise)
            nf = min(nf*2, self.params.ngf*16)
            h //= 2
            w //= 2
            d //= 2

        return out

