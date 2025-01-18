
import numpy as np
import matplotlib.pyplot as plt
# NOTE: these lines define global figure properties used for publication.
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') # display figures in vector format
plt.rcParams.update({'font.size':14}) # set global font size

A = np.random.randn(4,6)
# its SVD
U,s,Vt = np.linalg.svd(A)
# create Sigma from sigma's
S = np.zeros(np.shape(A))
np.fill_diagonal(S,s)
# show the matrices
_,axs = plt.subplots(1,4,figsize=(10,6))
axs[0].imshow(A,cmap='gray',aspect='equal')
axs[0].set_title('$\mathbf{A}$\nThe matrix')
axs[1].imshow(U,cmap='gray',aspect='equal')
axs[1].set_title('$\mathbf{U}$\n(left singular vects)')
axs[2].imshow(S,cmap='gray',aspect='equal')
axs[2].set_title('$\mathbf{\Sigma}$\n(singular vals)')
axs[3].imshow(Vt,cmap='gray',aspect='equal')
axs[3].set_title('$\mathbf{V}$\n(right singular vects)')
plt.tight_layout()
plt.savefig('Figure_14_02.png',dpi=300)
plt.show()