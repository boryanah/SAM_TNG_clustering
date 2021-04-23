import numpy as np
import matplotlib.pyplot as plt

# sim params
sam_dir = '/mnt/alan1/boryanah/SAM_subvolumes_TNG300/'
hydro_dir = '/mnt/gosling1/boryanah/TNG300/'
halo_type = 'tophat'
#halo_type = 'fof'
#star_str = ''
star_str = '_star'
vis_dir = "/mnt/gosling1/boryanah/TNG300/visuals/"+halo_type+"/"
halfrad_dir = "/mnt/gosling1/boryanah/TNG300/visuals/data_halfrad/"
inds_dir = "/home/boryanah/SAM/SAM_TNG_clustering/creating/visuals_"+halo_type+"/"
inds_low = np.load(inds_dir+"inds_low.npy").astype(int)
inds_high = np.load(inds_dir+"inds_high.npy").astype(int)

# position of the halo and virial radius
group_pos = np.load(hydro_dir+"GroupPos_fp.npy")
group_rvir = np.load(hydro_dir+"Group_R_Mean200_fp.npy")

# number of snapshots
n_snaps = 50

# number of gas particles per snapshot per halo
gas_nr_low = np.zeros((len(inds_low), n_snaps))
gas_nr_high = np.zeros((len(inds_high), n_snaps))
dm_nr_low = np.zeros((len(inds_low), n_snaps))
dm_nr_high = np.zeros((len(inds_high), n_snaps))

# how many (star) halfrad out
f_in = 2.
f_out = 20.


for i in range(len(inds_low)):
    print("i = ", i)
    snaps_low = np.load(halfrad_dir+"snaps_low_"+halo_type+"_"+str(inds_low[i])+".npy")
    snaps_high = np.load(halfrad_dir+"snaps_high_"+halo_type+"_"+str(inds_high[i])+".npy")

    halfrads_low = np.load(halfrad_dir+"halfrad"+star_str+"_low_"+halo_type+"_"+str(inds_low[i])+".npy")
    halfrads_high = np.load(halfrad_dir+"halfrad"+star_str+"_high_"+halo_type+"_"+str(inds_high[i])+".npy")

    for j in range(n_snaps):
        # snapshot and halfrad value
        snap_low = snaps_low[j]
        snap_high = snaps_high[j]
        halfrad_low = halfrads_low[j]
        halfrad_high = halfrads_high[j]
        
        # low indices
        gr_pos = group_pos[inds_low[i]]
        gas_pos = np.load(vis_dir+"gas_low_"+str(inds_low[i])+"_pos_"+str(snap_low)+".npy")
        dm_pos = np.load(vis_dir+"dm_low_"+str(inds_low[i])+"_pos_"+str(snap_low)+".npy")
        dist = np.sqrt(np.sum((gas_pos-gr_pos)**2, axis=1))
        gas_nr_low[i, j] = (gas_pos[(dist < halfrad_low*f_out) & (dist > halfrad_low*f_in)]).shape[0]
        dist = np.sqrt(np.sum((dm_pos-gr_pos)**2, axis=1))
        dm_nr_low[i, j] = (dm_pos[(dist < halfrad_low*f_out) & (dist > halfrad_low*f_in)]).shape[0]

        # high indices
        gr_pos = group_pos[inds_high[i]]
        gas_pos = np.load(vis_dir+"gas_high_"+str(inds_high[i])+"_pos_"+str(snap_high)+".npy")
        dm_pos = np.load(vis_dir+"dm_high_"+str(inds_high[i])+"_pos_"+str(snap_high)+".npy")
        dist = np.sqrt(np.sum((gas_pos-gr_pos)**2, axis=1))
        gas_nr_high[i, j] = (gas_pos[(dist < halfrad_high*f_out) & (dist > halfrad_high*f_in)]).shape[0]
        dist = np.sqrt(np.sum((dm_pos-gr_pos)**2, axis=1))
        dm_nr_high[i, j] = (dm_pos[(dist < halfrad_high*f_out) & (dist > halfrad_high*f_in)]).shape[0]

np.save("data/dm_nr_high_"+halo_type+".npy", dm_nr_high)
np.save("data/dm_nr_low_"+halo_type+".npy", dm_nr_low)
np.save("data/gas_nr_high_"+halo_type+".npy", gas_nr_high)
np.save("data/gas_nr_low_"+halo_type+".npy", gas_nr_low)
