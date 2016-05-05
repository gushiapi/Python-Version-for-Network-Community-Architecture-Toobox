
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import ncat as nt

#creating a pseudo network: 3 slices, 2 communities, one node switching back and forth
AijIn = np.zeros((5,5,4))

AijIn[:,:,0] = [[1,   0.6, 0,   0,   0],
[0.6, 1,   0,   0,   0],
[0,   0,   1,   0.6, 0.6],
[0,   0,   0.6, 1,   0.6],
[0,   0,   0.6, 0.6, 1]]

#%% 2nd slice
AijIn[:,:,1] = [[1,   0.6, 0.6,   0,   0],
[0.6, 1,   0.6,   0,   0],
[0.6, 0.6,  1,    0,   0],
[0,   0,    0,    1,   0.6],
[0,   0,    0,   0.6,   1]]


#%% 3rd slice
AijIn[:,:,2] = [[1,   0.6, 0,   0,   0],
[0.6, 1,   0,   0,   0],
[0,   0,   1,   0.6, 0.6],
[0,   0,   0.6, 1,   0.6],
[0,   0,   0.6, 0.6, 1]]

#%% 4th slice
AijIn[:,:,3] = [[1,   0.6, 0.6,   0,   0],
[0.6, 1,   0.6,   0,   0],
[0.6, 0.6,  1,    0,   0],
[0,   0,    0,    1,   0.6],
[0,   0,    0,   0.6,   1]]

plt.figure(figsize = (20,5))
for i in range(4):
    print(i)
    plt.subplot(1,4,i+1)
    plt.imshow(AijIn[:,:,i-1], cmap='seismic', interpolation = 'nearest')


#Set up the supra-adjacency
#AijIn = np.where(AijIn<0,0, AijIn)
gamma =1
omega=0.1
N=AijIn.shape[0]
T=AijIn.shape[2]

B = scipy.sparse.csr_matrix((N*T,N*T))    
twomu=0
for s in range(T):
    k=np.sum(AijIn[:,:,s],axis=0)   
    twom=np.sum(k)
    twomu=twomu+twom    
    indx=np.array(range(N))+(s*N)   
    B[indx[0]:indx[-1]+1,indx[0]:indx[-1]+1] = np.subtract(AijIn[:,:,s], (gamma * np.asmatrix(k).T * np.asmatrix(k)/twom))

twomu=twomu+2*omega*N*(T-1)

solve =  scipy.sparse.spdiags(np.ones((N*T,2)).T, np.array([-N,N]), N*T, T*N)
addthis = omega * solve
daddthis = addthis.todense()
B = B.todense() + daddthis


# Run community detection
nrep = 100;
commBCT = np.zeros((4, 5, nrep));

for rep in range(nrep):
    [S,Q] =  nt.genlouvain(B, ci=None, B='modularity', seed=None)
    commBCT[:,:,rep] = np.reshape(S, [4, 5]) 