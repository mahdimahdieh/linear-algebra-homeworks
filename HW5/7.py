import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for the subplots
import pandas as pd
import seaborn as sns
# NOTE: these lines define global figure properties used for publication.
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') # display figures in vector format
plt.rcParams.update({'font.size':14}) # set global font size

from skimage import io,color
url = 'https://berggasse19.org/wp-content/uploads/2015/05/stravinsky_picasso_wikipedia.png'
# import picture and downsample to 2D
strav = io.imread(url) / 255
#strav = color.rgb2gray(strav)
plt.figure(figsize=(8,8))
plt.imshow(strav,cmap='gray')
plt.title(f'Matrix size: {strav.shape}, rank:{np.linalg.matrix_rank(strav)}')
plt.show()

# SVD
U,s,Vt = np.linalg.svd(strav)
S = np.zeros_like(strav)
np.fill_diagonal(S,s)
# show scree plot
plt.figure(figsize=(12,4))
plt.plot(s[:30],'ks-',markersize=10)
plt.xlabel('Component index')
plt.ylabel('Singular value')
plt.title('Scree plot of Stravinsky picture')
plt.grid()
plt.show()

# Reconstruct based on first k layers
# number of components
k = 80
# reconstruction
stravRec = U[:,:k] @ S[:k,:k] @ Vt[:k,:]
# show the original, reconstructed, and error
_,axs = plt.subplots(1,3,figsize=(15,6))
axs[0].imshow(strav,cmap='gray',vmin=.1,vmax=.9)
axs[0].set_title('Original')
axs[1].imshow(stravRec,cmap='gray',vmin=.1,vmax=.9)
axs[1].set_title(f'Reconstructed (k={k}/{len(s)})')
axs[2].imshow((strav-stravRec)**2,cmap='gray',vmin=0,vmax=1e-1)
axs[2].set_title('Squared errors')
plt.tight_layout()
plt.savefig('Figure_15_10.png',dpi=300)
plt.show()