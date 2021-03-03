import os
import numpy as np
import matplotlib.pyplot as plt

import Corrfunc
from Corrfunc.theory.DD import DD
from Corrfunc.io import read_catalog
from Corrfunc.utils import convert_3d_counts_to_cf

def get_RR_norm(bins,Lbox,N1=1,N2=1):
    vol_all = 4./3*np.pi*bins**3
    vol_bins = vol_all[1:]-vol_all[:-1]
    n2 = N2/Lbox**3
    n2_bin = vol_bins*n2
    pairs = N1*n2_bin
    pairs /= (N1*N2)
    return pairs
                            
dtype = np.float32
def non_periodic_corrfunc(positions, weights, boxsize, return_bins, periodic=False):

    np.random.seed(0)

    bins = np.logspace(-1.,1.,21)
    bin_cents = 0.5*(bins[1:]+bins[:-1])

    X=positions[:,0]
    Y=positions[:,1]
    Z=positions[:,2]
    mean_w  = np.mean(weights)
    print("mean_w = ", mean_w)
    
    N = len(positions)
    nthreads = 16

    # Generate randoms on the box
    rand_N = 10*N
    rand_X = np.random.uniform(0, boxsize, rand_N).astype(dtype)
    rand_Y = np.random.uniform(0, boxsize, rand_N).astype(dtype)
    rand_Z = np.random.uniform(0, boxsize, rand_N).astype(dtype)
    rand_W = np.ones(rand_N, dtype=dtype)*mean_w

    
    # Auto pair counts in DD
    autocorr = 1
    DD_counts = DD(autocorr, nthreads, bins, X, Y, Z, weights1=weights, weight_type='pair_product', periodic=periodic, boxsize=boxsize)
        
    # Cross pair counts in DR
    autocorr = 0
    DR_counts = DD(autocorr, nthreads, bins, X, Y, Z, weights1=weights, weight_type='pair_product', X2=rand_X, Y2=rand_Y, Z2=rand_Z, periodic=periodic, weights2=rand_W)

    RD_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z, weights1=rand_W, weight_type='pair_product', X2=X, Y2=Y, Z2=Z, periodic=periodic, weights2=weights)



    # Auto pairs counts in RR
    autocorr = 1
    RR_counts = DD(autocorr, nthreads, bins, rand_X, rand_Y, rand_Z, weights1=rand_W, weight_type='pair_product', periodic=periodic, boxsize=boxsize)
    
    
    
    # All the pair counts are done, get the correlation function
    cf = convert_3d_counts_to_cf(N, N, rand_N, rand_N, DD_counts, DR_counts, RD_counts, RR_counts)
    

    '''
    DD_n = DD_counts['npairs']*DD_counts['weightavg']/N**2
    RR_n = RR_counts['npairs']*RR_counts['weightavg']/rand_N**2
    # works!
    #RR_n = get_RR_norm(bins, boxsize)
    #RR_n *= mean_w**2
    #cf = DD_n/RR_n-1
    DR_n = DR_counts['npairs']*DR_counts['weightavg']/(rand_N*N)
    RD_n = DR_counts['npairs']*DR_counts['weightavg']/(rand_N*N)
    #RD_n = RD_counts['npairs']/(rand_N*N)
    print(np.sum(DR_n-RD_n))
    cf = (DD_n-RD_n-DR_n+RR_n)/RR_n
    '''
    
    if return_bins == True:
        return cf, bin_cents
    
    elif return_bins == False:
        return cf

def xi_Corrfunc(positions, weights, boxLen, return_bins):
    '''
    gives positions, weights and size of box to pass to Corrfunc funtion
    '''
    bins = np.logspace(-1.,1.,21)
    bin_cents = 0.5*(bins[1:]+bins[:-1])

    xi = Corrfunc.theory.xi(X=positions[:,0], Y=positions[:,1], Z=positions[:,2], boxsize=boxLen,\
                            weights=weights, weight_type='pair_product', nthreads=16, binfile=bins)['xi']
    if return_bins==True:
        return xi, bin_cents
    
    elif return_bins==False:
        return xi
                            
pos_dir = os.path.expanduser("~/SAM/SAM_TNG_clustering/gm/data_pos/")
pos_gals = np.load(pos_dir+"/xyz_gals_hydro_12000_mstar.npy").astype(dtype)
pos_gals = pos_gals
#pos_gals_sam = np.load(pos_dir+"/xyz_gals_sam_12000_mstar.npy").astype(dtype)
N_gals = pos_gals.shape[0]
print("N_gals = ", N_gals)
#N_gals_sam = pos_gals_sam.shape[0]
Lbox = 205.
#weights = np.ones(N_gals).astype(dtype)

'''
weights = np.random.random(N_gals).astype(dtype)
weights[:3000] /= 1000.
weights[3000:6000] /= 10.
weights[6000:9000] *= 10
weights[9000:12000] *= 1000
'''
weights = np.load("sub.npy").astype(dtype)
wmin = 0.#1.e-1

'''
bins = np.linspace(0.01, weights.max(), 1000)
hist, bins = np.histogram(weights, bins=bins)
print(np.sum(hist))
bin_cents = (bins[1:] + bins[:-1])*0.5
plt.plot(bin_cents, hist)
plt.show()
quit()
'''

'''
boxLen=137
maskBox=pos_gals[:,0]<boxLen; maskBox*=pos_gals[:,1]<boxLen; maskBox*=pos_gals[:,2]<boxLen
pos_gals = pos_gals[maskBox]
weights = weights[maskBox]
Lbox = boxLen
'''

pos_gals = pos_gals[weights >= wmin]
weights = weights[weights >= wmin]
print(pos_gals.shape)
print(weights.min())


cf, bin_cents = xi_Corrfunc(pos_gals, weights, Lbox, True)
cf_np, bin_cents = non_periodic_corrfunc(pos_gals, weights, Lbox, True)

'''
plt.plot(bin_cents, cf*bin_cents**2, label="periodic")
plt.plot(bin_cents, cf_np*bin_cents**2, label="non-periodic")
plt.legend()
plt.show()
'''
plt.plot(bin_cents, np.ones(len(bin_cents)), 'k--')
plt.plot(bin_cents, cf/cf_np, label="periodic")
#plt.ylim([0.8, 1.2])
plt.show()
