import os
import sys

import numpy as np
import scipy.spatial as spatial
#from periodic_kdtree import PeriodicCKDTree # python2
import itertools

# do the computation for the hydro galaxies or SAM?
want_true = 0

# what are we displaying
opt = 'norm'#sys.argv[1]#"partial_fenv"#"partial_s2r"#"shuff"#"spin"#"shuff"#"conc"#"vani"#"shuff"#"env"#"partial_vani""partial_s2r""vdisp"

Lbox = 205.; N_dim = 128 # in Mpc/h
#Lbox = 75.; N_dim = 64 # in Mpc/h
bounds = np.array([Lbox,Lbox,Lbox])
gr_size = Lbox/N_dim
box_cents = np.linspace(0,Lbox,N_dim+1)
box_cents = .5*(box_cents[1:]+box_cents[:-1])
all_cents = np.array(list(itertools.product(box_cents, repeat=3)))

# load galaxies
pos_dir = os.path.expanduser("~/SAM/SAM_TNG_clustering/gm/data_pos/")
pos_g = np.load(pos_dir+"/xyz_gals_hydro_12000_mstar.npy").astype(np.float32)
pos_g_opt = np.load(pos_dir+"/xyz_gals_sam_12000_mstar.npy").astype(np.float32)
N_g = pos_g.shape[0]
N_g_opt = pos_g.shape[0]
print("Number of gals = ",N_g)

# build tree for each of the two galaxy populations
tree_g = spatial.cKDTree(pos_g, boxsize=Lbox)
tree_g_opt = spatial.cKDTree(pos_g_opt, boxsize=Lbox)
#tree_g = PeriodicCKDTree(bounds, pos_g)
#tree_g_opt = PeriodicCKDTree(bounds, pos_g_opt)
print("Built the trees")

if want_true:
    void_g = np.zeros(N_dim**3)
void_g_opt = np.zeros(N_dim**3)

import time
t1 = time.time()
# query the tree for each box_center at each radius
for i_xyz in range(N_dim**3):
    box_center = all_cents[i_xyz]

    # blah query and distance
    if want_true:
        dist_g = np.max(tree_g.query(box_center, k=3)[0])
        void_g[i_xyz] = dist_g
    dist_g_opt = np.max(tree_g_opt.query(box_center, k=3)[0])
    void_g_opt[i_xyz] = dist_g_opt

if want_true:
    np.save("data/void_true.npy",void_g)
np.save("data/void_"+opt+".npy",void_g_opt)
t2 = time.time()
print("Time elapsed = ",t2-t1)
