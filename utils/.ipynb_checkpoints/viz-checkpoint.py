import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from scipy import signal


def viz_fields(flist, output):
    
    '''
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
    ''' 
    
    pred, tar = flist
    
    if output == 'temp':
        pred = np.exp(8*(pred[0][:,:,16]+1.5))
        tar = np.exp(8*(tar[0][:,:,16]+1.5))
        thisNorm = LogNorm(vmin=1e3, vmax=1e+7)
        bins=np.logspace(3, 6, 50)
    elif output == 'flux':
        pred = pred[0][:,:,16]
        tar = tar[0][:,:,16]
        thisNorm = Normalize(vmin=0, vmax=1)
        bins=np.linspace(0, 1, num=50, endpoint=True)
    else:
        pred = np.exp(14.*pred[0][:,:,16])
        tar = np.exp(14.*tar[0][:,:,16])
        thisNorm = LogNorm(vmin=1e-1, vmax=1e+3)
        bins=np.logspace(-2, 4, 50)
        
    
    #thisNorm = LogNorm(vmin=np.log(tar.min()), vmax=np.log(tar.max()))
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12,12))    
    plt.subplot(2,2,1)
    plt.imshow(pred, cmap='magma', norm=thisNorm)
    plt.title('Generated')
    plt.subplot(2,2,2)
    plt.imshow(tar, cmap='magma', norm=thisNorm)
    plt.title('Truth')
    plt.subplot(2,2,3)
    if output == 'flux': bins = plt.hist(tar.flatten(), bins=bins, density=True, histtype='step', log=False, color='black')[1]
    else: bins = plt.hist(tar.flatten(), bins=bins, density=True, histtype='step', log=True, color='black')[1]
    plt.hist(pred.flatten(), bins=bins, range=None, density=True,  histtype='step', log=True, color='grey')
    if not output == 'flux': plt.xscale('log')
    plt.subplot(2,2,4)
    f, Pxx_den = signal.welch(tar, fs=1, nperseg=128)
    Pk = np.mean(Pxx_den, axis=(0))
    #f = f*40
    plt.loglog(f,Pk,c='black')
    f, Pxx_den = signal.welch(pred, fs=1, nperseg=128)
    Pk = np.mean(Pxx_den, axis=(0))
    #f = f*40
    plt.loglog(f,Pk,c='grey')
    
    plt.tight_layout()
    return fig
