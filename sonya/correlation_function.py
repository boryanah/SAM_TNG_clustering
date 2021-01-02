import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

np.random.seed(0)

def get_corrfunc(pos,bins,Lbox,periodic=True):
    # generate N*factor random particles
    N = pos.shape[0]
    factor = 30
    pos_r = np.random.random((N*factor, 3))*Lbox
    
    # generate the trees
    tree = cKDTree(pos,Lbox)
    tree_r = cKDTree(pos_r,Lbox)

    # count pairs
    cumulative = False
    DD = tree.count_neighbors(tree,bins,cumulative=cumulative)[1:]
    RR = tree_r.count_neighbors(tree_r,bins,cumulative=cumulative)[1:]
    DR = tree.count_neighbors(tree_r,bins,cumulative=cumulative)[1:]

    # compute the correlation function
    f = N*1./pos_r.shape[0]
    if periodic:
        # Landy-Szalay method for periodic
        corr = DD/(RR*f**2)-1.
    else:
        # Landy-Szalay method for non-periodic
        corr = DD-2*f*DR+f**2*RR
        corr /= (f**2*RR)

    return corr

def main():
    Lbox = 75.
    # load the particles or galaxies
    pos_g = np.random.random((2000, 3))*Lbox

    # compute correlation function
    bins = np.logspace(-1, 1, 21)
    bin_centers = .5*(bins[1:]+bins[:-1])
    corr = get_corrfunc(pos_g,bins,Lbox)

    # show xi*r^2 vs. r
    plt.plot(bin_centers,corr*bin_centers**2)
    plt.xscale('log')
    # distance in Mpc/h
    plt.xlabel('r [Mpc/h]')
    # correlation function xi multiplied with the distance squared
    plt.ylabel('xi * r**2')
    plt.show()
    '''
    # for comparison with corrfunc
    import Corrfunc

    res = Corrfunc.theory.xi(X=pos_g[:,0],Y=pos_g[:,1],Z=pos_g[:,2],boxsize=Lbox,nthreads=16,binfile=bins)
    Corr = res['xi']

    plt.plot(bin_centers,corr,label="Mine")
    plt.plot(bin_centers,Corr,label="Corrfunc")
    plt.xscale('log')
    plt.legend()
    plt.show()
    '''

