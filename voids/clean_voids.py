import numpy as np
import scipy.spatial as spatial
from periodic_kdtree import PeriodicCKDTree
import itertools
import sys
import time

# what are we displaying
proxy = "m200m"
opt = sys.argv[1]#"partial_fenv"#"partial_s2r"#"shuff"#"spin"#"shuff"#"conc"#"vani"#"shuff"#"env"#"partial_vani""partial_s2r""vdisp"

if len(sys.argv) > 2:
    want_true = int(sys.argv[2])
else:
    # do the computation for the true galaxies?
    want_true = 0


# parameter choices
Lbox = 205. # in Mpc/h
bounds = np.array([Lbox,Lbox,Lbox])
N_dim = 128
gr_size = Lbox/N_dim
box_cents = np.linspace(0,Lbox,N_dim+1)
box_cents = .5*(box_cents[1:]+box_cents[:-1])
all_cents = np.array(list(itertools.product(box_cents, repeat=3)))

# load all overlapping voids
void_g = np.load("data_void/void_true.npy")
void_g_opt = np.load("data_void/void_"+proxy+"_"+opt+".npy")

# select how many of the largest voids we want
n_top = 1000000

# indices sorted
i_sort_opt_top = np.argsort(void_g_opt)[::-1]
i_sort_opt_top = i_sort_opt_top[:n_top]
cents_opt_top = all_cents[i_sort_opt_top]
voids_opt_top = void_g_opt[i_sort_opt_top]

i_sort_top = np.argsort(void_g)[::-1]
i_sort_top = i_sort_top[:n_top]
cents_top = all_cents[i_sort_top]
voids_top = void_g[i_sort_top]

# build tree for each of the two galaxy populations
#tree_g = spatial.cKDTree(pos_g)
#tree_g_opt = spatial.cKDTree(pos_g_opt)
tree_g = PeriodicCKDTree(bounds, cents_top)
tree_g_opt = PeriodicCKDTree(bounds, cents_opt_top)
print("Built the trees")


def prune_voids(tree,cent_top,void_top):
    i_void_tru = []
    i_void_fal = np.array([])
    void_no = 0
    # query the tree for each box_center at void radius
    for i in range(n_top):
        if i in i_void_fal: continue
        void_no += 1
        if void_no == 1000: print("Reached 1000th void")
        void_center = cent_top[i]
        void_radius = void_top[i]

        # blah query and distance
        i_inside = np.array(tree.query_ball_point(void_center, void_radius))
        i_void_fal = np.hstack((i_void_fal,i_inside))
        i_void_fal = np.unique(i_void_fal)
        i_void_tru.append(i)
    print("Last void has radius = ",void_radius)
    i_void_tru = np.array(i_void_tru)
    sizes_void_tru = cent_top[i_void_tru]
    cents_void_tru = void_top[i_void_tru].reshape(-1,1)
    voids_tru = np.hstack((cents_void_tru,sizes_void_tru))
    return voids_tru

t1 = time.time()
voids_opt = prune_voids(tree_g_opt,cents_opt_top,voids_opt_top)
print("number of voids = ",voids_opt.shape[0])
np.save("data_clean/clean_void_"+proxy+"_"+opt+".npy",voids_opt)
t2 = time.time()
print("Time elapsed = ",t2-t1)

if want_true:
    t1 = time.time()
    voids = prune_voids(tree_g,cents_top,voids_top)
    print("number of voids in true = ",voids.shape[0])
    t2 = time.time()
    np.save("data_clean/clean_void_true.npy",voids)
    print("Time elapsed = ",t2-t1)
