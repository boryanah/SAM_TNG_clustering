import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# sim params
sam_dir = '/mnt/alan1/boryanah/SAM_subvolumes_TNG300/'
hydro_dir = '/mnt/gosling1/boryanah/TNG300/'
str_snap = ''
Lbox = 205.

# loading SAM
sat_type = np.load(sam_dir+'GalpropSatType'+str_snap+'.npy')
halo_pos = np.load(sam_dir+'GalpropPos'+str_snap+'.npy')[sat_type == 0]
halo_mvir = np.load(sam_dir+'GalpropMhalo'+str_snap+'.npy')[sat_type == 0]
halo_rvir = np.load(sam_dir+'GalpropRhalo'+str_snap+'.npy')[sat_type == 0]
N_h = halo_pos.shape[0]

# sort all arrays in order of halo mass
i_sort = np.argsort(halo_mvir)[::-1]
halo_mvir = halo_mvir[i_sort]
halo_pos = halo_pos[i_sort]
halo_rvir = halo_rvir[i_sort]

# loading TNG 
SubhaloMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp'+str_snap+'.npy')[:, 4]*1.e10
SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp'+str_snap+'.npy')/1.e3
choice = SubhaloMstar_fp > 0.
points_ref = SubhaloPos_fp[choice]
points_ref %= Lbox
inds_ref = np.arange(len(SubhaloMstar_fp), dtype=int)[choice]
N_g = points_ref.shape[0]
print("number of galaxies = ", N_g)

# form a KD-tree from the subhalo positions
tree = cKDTree(points_ref, boxsize=Lbox)
print("formed tree")

# factor to search around each halo
fac = 1.

# for each ROCKSTAR halo, you have which TNG subhalo corresponds to it
idx = tree.query_ball_point(halo_pos, fac*halo_rvir)
# this is the number per ROCKSTAR halo
#length = tree.query_ball_point(halo_pos, fac*halo_rvir, return_length=True)
# faster
length = np.array([(np.array(idx[i])).shape[0] for i in range(idx.shape[0])])


# stack together the indices into a long-ass array
all_idx = np.hstack((idx)).astype(int)
assert np.sum(length) == len(all_idx), "differing lengths"

# index for each ROCKSTAR halo
halo_inds = np.arange(halo_rvir.shape[0], dtype=int)
all_halo_inds = np.repeat(halo_inds, length)

# unique subhalo (modified) indices
all_idx_uni, inds = np.unique(all_idx, return_index=True)
# corresponding halo indices (ready to use) for the unique subhalos (modified)
all_halo_inds_uni = all_halo_inds[inds]
#halo_counts = np.zeros(halo_rvir.shape[0], dtype=int)

# for each TNG subhalo, what is its ROCKSTAR host
SubhaloHaloNr = np.zeros(len(SubhaloMstar_fp), dtype=int)-1
(SubhaloHaloNr[choice])[all_idx_uni] = all_halo_inds_uni
np.save("SubhaloHaloNr_fp.npy", SubhaloHaloNr)
#dic = dict(zip(halo_inds, idx))
