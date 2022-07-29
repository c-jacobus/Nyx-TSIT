import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import FADEResnetBlock as FADEResnetBlock
from models.networks.stream import Stream as Stream
from models.networks.stream import NoiseStream as NoiseStream
from models.networks.AdaIN.function import adaptive_instance_normalization as FAdaIN

class TSITGenerator(BaseNetwork):

    def __init__(self, params):
        super().__init__()
        self.params = params
        nf = params.ngf
        self.ppad = False
        self.FADEResnetBlock = FADEResnetBlock
        
        self.sw, self.sh, self.sd, self.n_stages = self.compute_latent_vector_size()
        self.content_stream = Stream(self.params, reshape_size=(self.sh*(2**self.n_stages), self.sw*(2**self.n_stages), self.sd*(2**self.n_stages)))
        self.style_stream = Stream(self.params) if not self.params.no_ss else None
        self.noise_stream = NoiseStream(self.params) if self.params.additive_noise else None

        if params.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(params.z_dim, 16 * nf * self.sw * self.sh * self.sd)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled input instead of random z
            self.nz = self.params.z_dim if not self.params.downsamp else self.params.input_nc
            convpad = 1
            self.fc = nn.Conv3d(self.nz, 16 * nf, 3, padding=convpad)

        self.head_0 = self.FADEResnetBlock(16 * nf, 16 * nf, params)

        self.G_middle_0 = self.FADEResnetBlock(16 * nf, 16 * nf, params)
        self.G_middle_1 = self.FADEResnetBlock(16 * nf, 16 * nf, params)

        self.up_0 = self.FADEResnetBlock(16 * nf, 8 * nf, params)
        self.up_1 = self.FADEResnetBlock(8 * nf, 4 * nf, params)
        self.up_2 = self.FADEResnetBlock(4 * nf, 2 * nf, params)
        self.up_3 = self.FADEResnetBlock(2 * nf, 1 * nf, params)

        final_nc = nf

        self.up_4 = self.FADEResnetBlock(1 * nf, 1 * nf, params) if self.params.num_upsampling_blocks == 8 else None
        
        self.conv_img = nn.Conv3d(final_nc, self.params.output_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2.)

    def compute_latent_vector_size(self):
        num_blocks = self.params.num_upsampling_blocks
        #print(f'num_upsampling_blocks = {num_blocks}')
        #print(f'data_size = {self.params.data_size}')
        sw, sh, sd = self.params.data_size // (2**num_blocks), self.params.data_size // (2**num_blocks), self.params.data_size // (2**num_blocks)
        # sw, sh, sd = self.params.img_size[0] // (2**num_blocks), self.params.img_size[1] // (2**num_blocks), self.params.img_size[2] // (2**num_blocks)
        return sw, sh, sd, num_blocks

    def fadain_alpha(self, content_feat, style_feat, alpha=1.0, c_mask=None, s_mask=None):
        # FAdaIN performs AdaIN on the multi-scale feature representations
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        t = alpha * t + (1 - alpha) * content_feat
        return t

    def forward(self, input, real, z=None):
        content = input
        #print(f'Generator content = {content.size()}')
        #print(f'x dims = {self.sw}, {self.sh}, {self.sd}')
        style =  real
        ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7 = self.content_stream(content)
        sft0, sft1, sft2, sft3, sft4, sft5, sft6, sft7 = self.style_stream(style) if not self.params.no_ss else [None] * 8
        nft0, nft1, nft2, nft3, nft4, nft5, nft6, nft7 = self.noise_stream(style) if self.params.additive_noise else [None] * 8
        if self.params.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(content.size(0), self.params.z_dim,
                                dtype=torch.float32, device=content.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.params.ngf, self.sw, self.sh, self.sd)
        else:
            if self.params.downsamp:
                # following SPADE, downsample segmap and run convolution for SIS
                x = F.interpolate(content, size=(self.sw, self.sh, self.sd))
            else:
                # sample random noise
                x = torch.randn(content.size(0), 3, self.sw, self.sh, self.sd, dtype=torch.float32, device=content.get_device())
            x = self.fc(x)
            
        #print(f'content stream: {ft0} {ft1} {ft2} {ft3} {ft4} {ft5} {ft6} {ft7}')
        #print(f'style stream: {sft0} {sft1} {sft2} {sft3} {sft4} {sft5} {sft6} {sft7}')
        #print(f'noise stream: {nft0} {nft1} {nft2} {nft3} {nft4} {nft5} {nft6} {nft7}')
        '''
        x = self.fadain_alpha(x, sft7, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft7 if self.params.additive_noise else x
        x = self.head_0(x, ft7)

        x = self.up(x)
        '''
        
        
        

        '''
        if self.params.num_upsampling_blocks == 7 or \
           self.params.num_upsampling_blocks == 8:
            x = self.up(x)
        '''
        if self.params.num_upsampling_blocks >= 6:
            x = self.fadain_alpha(x, sft6, alpha=self.params.alpha) if not self.params.no_ss else x
            x = x + nft6 if self.params.additive_noise else x 
            x = self.G_middle_0(x, ft6)
            x = self.up(x)
            
        x = self.fadain_alpha(x, sft5, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft5 if self.params.additive_noise else x
        x = self.G_middle_1(x, ft5)

        x = self.up(x)
        x = self.fadain_alpha(x, sft4, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft4 if self.params.additive_noise else x
        x = self.up_0(x, ft4)

        x = self.up(x)
        x = self.fadain_alpha(x, sft3, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft3 if self.params.additive_noise else x
        x = self.up_1(x, ft3)
        
        x = self.up(x)
        x = self.fadain_alpha(x, sft2, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft2 if self.params.additive_noise else x
        x = self.up_2(x, ft2)
        
        x = self.up(x)
        x = self.fadain_alpha(x, sft1, alpha=self.params.alpha) if not self.params.no_ss else x
        x = x + nft1 if self.params.additive_noise else x
        x = self.up_3(x, ft1)

        x = self.up(x)
        if self.params.num_upsampling_blocks == 8:
            ft0 = self.up(ft0)
            x = self.fadain_alpha(x, sft0, alpha=self.params.alpha) if not self.params.no_ss else x
            x = x + nft0 if self.params.additive_noise else x
            x = self.up_4(x, ft0)
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if self.params.tanh:
            x = F.tanh(x) #was relu
        return x