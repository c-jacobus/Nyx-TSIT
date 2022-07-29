import re
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a standard normalization function
def get_norm_layer(params, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm3d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm3d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates FADE normalization layer based on the given configuration
class FADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, ppad=False):
        super().__init__()
        assert config_text.startswith('fade')
                
        parsed = re.search('fade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in FADE'
                             % param_free_norm_type)
            
        
        pw = ks // 2 + 1 #was not trained with +1
        convpad =  pw
        
        '''
        print('label_nc = {}'.format(label_nc))
        print('norm_nc = {}'.format(norm_nc))
        print('ks = {}'.format(ks))
        print('convpad = {}'.format(convpad))
        '''
        
        self.mlp_gamma = nn.Conv3d(label_nc, norm_nc, kernel_size=ks, padding=convpad)
        self.mlp_beta = nn.Conv3d(label_nc, norm_nc, kernel_size=ks, padding=convpad)

    def forward(self, x, feat):
        # Step 1. generate parameter-free normalized activations
        
        #print('x = {}'.format(x.size()))
        #print('feat = {}'.format(feat.size()))
        normalized = self.param_free_norm(x)
        
        #print('normalized = {}'.format(normalized.size()))
        
        
        #pfeat = nn.Identity(feat)
        
        # Step 2. produce scale and bias conditioned on feature map
        gamma = self.mlp_gamma(feat)
        beta = self.mlp_beta(feat)
        
        #print('gamma = {}'.format(gamma.size()))
        #print('beta = {}'.format(beta.size()))
        
        # Step 3. apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
