import numpy as np
import matplotlib.pyplot as plt
from halotools.mock_observables import mean_delta_sigma
from halotools.utils import randomly_downsample_data

import plotparams
plotparams.buba()
import sys
import os

dir_part = "/mnt/gosling1/boryanah/TNG300/"
part_fn = "parts_position_tng300-3_99.npy"
proxy = "m200m"
ext1 = "data_2dhod_pos"
ext2 = "data_2dhod_peak"

opt = sys.argv[1]#"shuff"#"spin"#"shuff"#"conc"#"vani"#"shuff"#"env"

Lbox = 205. # in Mpc/h
period = np.array([Lbox,Lbox,Lbox])
N_parts = 625**3
N_bin = 11
rp_bins = np.logspace(-0.7, 1.3, N_bin)
rp_mids = .5*(rp_bins[1:]+rp_bins[1:])
pcle_mass = 3.03e9 # in Msun/h
n_thread = 16
N_parts_down = N_parts//40#used to be 4
down_parts_fname = "../pos_parts_down_"+str(N_parts_down)+".npy"


test = False
if os.path.isfile(down_parts_fname):
    pos_parts = np.load(down_parts_fname)
else:
    if test:
        Npts = int(1e5)
        x = np.random.uniform(0, Lbox, Npts)
        y = np.random.uniform(0, Lbox, Npts)
        z = np.random.uniform(0, Lbox, Npts)
        pos_parts = np.vstack((x, y, z)).T
    else:
        pos_parts = np.load(dir_part+part_fn)/1000. # in Mpc/h
    pos_parts = randomly_downsample_data(pos_parts, N_parts_down)
    print(pos_parts.shape[0])
    np.save(down_parts_fname,pos_parts)

if test:
    Npts = int(1.2e4)
    x = np.random.uniform(0, Lbox, Npts)
    y = np.random.uniform(0, Lbox, Npts)
    z = np.random.uniform(0, Lbox, Npts)
    pos_gals = np.vstack((x, y, z)).T
else:
    pos_gals = np.load("../"+ext1+"/true_gals.npy")

if test:
    Npts = int(1.2e4)
    x = np.random.uniform(0, Lbox, Npts)
    y = np.random.uniform(0, Lbox, Npts)
    z = np.random.uniform(0, Lbox, Npts)
    pos_gals_opt = np.vstack((x, y, z)).T
else:
    pos_gals_opt = np.load("../"+ext2+"/"+proxy+"_"+opt+"_gals.npy")

down_fac = N_parts/N_parts_down
pcle_mass *= down_fac

N_gals = pos_gals.shape[0]
N_gals_opt = pos_gals_opt.shape[0]

print(N_gals)
print(N_gals_opt)


N_dim = 3

Rat_Delta = np.zeros((N_bin-1,N_dim**3))
Delta_Sigma = np.zeros((N_bin-1,N_dim**3))
Delta_Sigma_opt = np.zeros((N_bin-1,N_dim**3))
for i_x in range(N_dim):
    for i_y in range(N_dim):
        for i_z in range(N_dim):
            print(i_x,i_y,i_z)
            
            xyz_jack = pos_gals.copy()
            xyz_opt_jack = pos_gals_opt.copy()
            xyz_m_jack = pos_parts.copy()

            xyz = np.array([i_x,i_y,i_z],dtype=int)
            size = Lbox/N_dim

            bool_arr = np.prod((xyz == (xyz_jack/size).astype(int)),axis=1).astype(bool)
            xyz_jack[bool_arr] = np.array([0.,0.,0.])
            this_chunk = np.sum(xyz_jack,axis=1)!=0.
            xyz_jack = xyz_jack[this_chunk]

            bool_arr = np.prod((xyz == (xyz_opt_jack/size).astype(int)),axis=1).astype(bool)
            xyz_opt_jack[bool_arr] = np.array([0.,0.,0.])
            this_chunk = np.sum(xyz_opt_jack,axis=1)!=0.
            xyz_opt_jack = xyz_opt_jack[this_chunk]

            bool_arr = np.prod((xyz == (xyz_m_jack/size).astype(int)),axis=1).astype(bool)
            xyz_m_jack[bool_arr] = np.array([0.,0.,0.])
            this_chunk = np.sum(xyz_m_jack,axis=1)!=0.
            xyz_m_jack = xyz_m_jack[this_chunk]
            
            ds = mean_delta_sigma(xyz_jack, xyz_m_jack, pcle_mass, rp_bins, period, num_threads=n_thread)
            ds_opt = mean_delta_sigma(xyz_opt_jack, xyz_m_jack, pcle_mass, rp_bins, period, num_threads=n_thread)
            ds_rat = ds_opt/ds
            
            Delta_Sigma[:,i_x+N_dim*i_y+N_dim**2*i_z] = ds
            Delta_Sigma_opt[:,i_x+N_dim*i_y+N_dim**2*i_z] = ds_opt
            Rat_Delta[:,i_x+N_dim*i_y+N_dim**2*i_z] = ds_rat

Delta_mean = np.mean(Delta_Sigma,axis=1)
Delta_err = np.sqrt(N_dim**3-1)*np.std(Delta_Sigma,axis=1)
Delta_opt_mean = np.mean(Delta_Sigma_opt,axis=1)
Delta_opt_err = np.sqrt(N_dim**3-1)*np.std(Delta_Sigma_opt,axis=1)
Rat_mean = np.mean(Rat_Delta,axis=1)
Rat_err = np.sqrt(N_dim**3-1)*np.std(Rat_Delta,axis=1)

np.save("../data_jack/rp_mids.npy",rp_mids)
np.save("../data_jack/ds_true_mean.npy",Delta_mean)
np.save("../data_jack/ds_true_err.npy",Delta_err)
np.save("../data_jack/ds_"+proxy+"_"+opt+"_mean.npy",Delta_opt_mean)
np.save("../data_jack/ds_"+proxy+"_"+opt+"_err.npy",Delta_opt_err)
np.save("../data_jack/ds_rat_"+proxy+"_"+opt+"_mean.npy",Rat_mean)
np.save("../data_jack/ds_rat_"+proxy+"_"+opt+"_err.npy",Rat_err)
