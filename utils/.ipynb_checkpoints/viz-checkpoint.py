import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def viz_fields(flist):
    pred, tar = flist
    pred = pred[0][:,:,16]
    tar = tar[0][:,:,16]
    top = tar.max()
    bot = tar.min()
    f = plt.figure(figsize=(12,6))    
    plt.subplot(1,2,1)
    plt.imshow(pred, cmap='magma', norm=Normalize(bot, top))
    plt.title('Generated')
    plt.subplot(1,2,2)
    plt.imshow(tar, cmap='magma', norm=Normalize(bot, top))
    plt.title('Truth')
        
    plt.tight_layout()
    return f
