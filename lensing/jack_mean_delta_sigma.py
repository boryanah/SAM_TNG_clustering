import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from halotools.mock_observables import mean_delta_sigma
from halotools.utils import randomly_downsample_data

import plotparams
plotparams.buba()


Lbox = 205. # in Mpc/h
#Lbox = 75. # in Mpc/h
if np.abs(Lbox - 75) < 1.e-5: str_tng = "_tng100-3"; dir_part = "/mnt/gosling1/boryanah/TNG100/"; part_fn = "pos_parts"+str_tng+".npy"
elif np.abs(Lbox - 205) < 1.e-5: str_tng = "_tng300-3"; dir_part = "/mnt/gosling1/boryanah/TNG300/"; part_fn = "pos_parts"+str_tng+"_99.npy"
period = np.array([Lbox,Lbox,Lbox])
if 'tng300-2' in part_fn: N_parts = 1250**3; pcle_mass = 3.78173109286472e8
elif 'tng300-3' in part_fn: N_parts = 625**3; pcle_mass = 3.02538487429177e9
elif 'tng100-3' in part_fn: N_parts = 455**3; pcle_mass = 3.83980084934323.e8 # Msun/h

# bins
N_bin = 11
rp_bins = np.logspace(-0.7, 1.3, N_bin)
rp_mids = .5*(rp_bins[1:]+rp_bins[1:])
n_thread = 16

# downsampling specs 
down = 5000#4000 #500, 4000 for tng300-2 # 5000 for tng300-3
N_parts_down = N_parts//down
down_parts_fname = dir_part+"pos_parts_down_"+str(down)+str_tng+".npy"

if os.path.isfile(down_parts_fname):
    pos_parts = np.load(down_parts_fname)
    print("Load from existing")
else:
    pos_parts = np.load(dir_part+part_fn)/1000. # in Mpc/h
    pos_parts = randomly_downsample_data(pos_parts, N_parts_down)
    print(pos_parts.shape[0])
    np.save(down_parts_fname,pos_parts)

# load the galaxies
pos_dir = os.path.expanduser("~/SAM/SAM_TNG_clustering/gm/data_pos/")
pos_gals = np.load(pos_dir+"/xyz_gals_hydro_12000_mstar.npy").astype(np.float32)
pos_gals_sam = np.load(pos_dir+"/xyz_gals_sam_12000_mstar.npy").astype(np.float32)
N_gals = pos_gals.shape[0]
N_gals_sam = pos_gals_sam.shape[0]
print(N_gals)
print(N_gals_sam)

# get the mass of the particles
pcle_mass *= down


N_dim = 3
Rat_Delta = np.zeros((N_bin-1,N_dim**3))
Delta_Sigma = np.zeros((N_bin-1,N_dim**3))
Delta_Sigma_sam = np.zeros((N_bin-1,N_dim**3))
for i_x in range(N_dim):
    for i_y in range(N_dim):
        for i_z in range(N_dim):
            print(i_x,i_y,i_z)
            
            xyz_jack = pos_gals.copy()
            xyz_sam_jack = pos_gals_sam.copy()
            xyz_m_jack = pos_parts.copy()

            xyz = np.array([i_x,i_y,i_z],dtype=int)
            size = Lbox/N_dim

            bool_arr = np.prod((xyz == (xyz_jack/size).astype(int)),axis=1).astype(bool)
            xyz_jack[bool_arr] = np.array([0.,0.,0.])
            this_chunk = np.sum(xyz_jack,axis=1)!=0.
            xyz_jack = xyz_jack[this_chunk]

            bool_arr = np.prod((xyz == (xyz_sam_jack/size).astype(int)),axis=1).astype(bool)
            xyz_sam_jack[bool_arr] = np.array([0.,0.,0.])
            this_chunk = np.sum(xyz_sam_jack,axis=1)!=0.
            xyz_sam_jack = xyz_sam_jack[this_chunk]

            bool_arr = np.prod((xyz == (xyz_m_jack/size).astype(int)),axis=1).astype(bool)
            xyz_m_jack[bool_arr] = np.array([0.,0.,0.])
            this_chunk = np.sum(xyz_m_jack,axis=1)!=0.
            xyz_m_jack = xyz_m_jack[this_chunk]
            
            ds = mean_delta_sigma(xyz_jack, xyz_m_jack, pcle_mass, rp_bins, period, num_threads=n_thread)
            ds_sam = mean_delta_sigma(xyz_sam_jack, xyz_m_jack, pcle_mass, rp_bins, period, num_threads=n_thread)
            ds_rat = ds_sam/ds
            
            Delta_Sigma[:,i_x+N_dim*i_y+N_dim**2*i_z] = ds
            Delta_Sigma_sam[:,i_x+N_dim*i_y+N_dim**2*i_z] = ds_sam
            Rat_Delta[:,i_x+N_dim*i_y+N_dim**2*i_z] = ds_rat


Delta_mean = np.mean(Delta_Sigma,axis=1)
Delta_err = np.sqrt(N_dim**3-1)*np.std(Delta_Sigma,axis=1)
Delta_sam_mean = np.mean(Delta_Sigma_sam,axis=1)
Delta_sam_err = np.sqrt(N_dim**3-1)*np.std(Delta_Sigma_sam,axis=1)
Rat_mean = np.mean(Rat_Delta,axis=1)
Rat_err = np.sqrt(N_dim**3-1)*np.std(Rat_Delta,axis=1)

print(Delta_mean, Delta_sam_mean)
np.save("data/rp_mids.npy",rp_mids)
np.save("data/ds_hydro_mean.npy",Delta_mean)
np.save("data/ds_hydro_err.npy",Delta_err)
np.save("data/ds_sam_mean.npy",Delta_sam_mean)
np.save("data/ds_sam_err.npy",Delta_sam_err)
np.save("data/ds_rat_norm_mean.npy",Rat_mean)
np.save("data/ds_rat_norm_err.npy",Rat_err)
