import sys
import os

import numpy as np
import scipy.spatial as spatial
import itertools
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# what are we displaying
sam = 'norm'#"partial_fenv"#"partial_s2r"#"shuff"#"spin"#"shuff"#"conc"#"vani"#"shuff"#"env"#"partial_vani""partial_s2r""vdisp"

Lbox = 205. # in Mpc/h
#Lbox = 75. # in Mpc/h
if np.abs(Lbox - 75) < 1.e-5: str_tng = "_tng100-3"; dir_part = "/mnt/gosling1/boryanah/TNG100/"; part_fn = "pos_parts"+str_tng+".npy"
elif np.abs(Lbox - 205) < 1.e-5: str_tng = "_tng300-2"; dir_part = "/mnt/gosling1/boryanah/TNG300/"; part_fn = "pos_parts"+str_tng+"_99.npy"#tng300-3#part_fn = "parts_position_tng300-3_99.npy"#tng300-3
n_thread = 16
periodic = True
if 'tng300-2' in part_fn: N_m = 1250**3; N_dim = 256
elif 'tng300-3' in part_fn: N_m = 625**3; N_dim = 256
elif 'tng100-3' in part_fn: N_m = 455**3; N_dim = 128
gr_size = Lbox/N_dim
Rs = np.linspace(3,8,6)/gr_size # vajno!!!

# load them galaxies
pos_dir = os.path.expanduser("~/SAM/SAM_TNG_clustering/gm/data_pos/")
pos_g = np.load(pos_dir+"/xyz_gals_hydro_12000_mstar.npy").astype(np.float32)
pos_g_sam = np.load(pos_dir+"/xyz_gals_sam_12000_mstar.npy").astype(np.float32)
N_g = pos_g.shape[0]
N_g_sam = pos_g_sam.shape[0]
print("Number of gals = ",N_g_sam)

def get_density(pos):
    g_x = pos[:,0]
    g_y = pos[:,1]
    g_z = pos[:,2]
    D, edges = np.histogramdd(np.transpose([g_x,g_y,g_z]),bins=N_dim,range=[[0,Lbox],[0,Lbox],[0,Lbox]])
    D_avg = N_g*1./N_dim**3
    D /= D_avg
    D -= 1.
    return D

D_g = get_density(pos_g)
D_g_sam = get_density(pos_g_sam)


def Wg(k2, r):
    return np.exp(-k2*r*r/2.)


def get_smooth_density(D,rsmo=4.):
    karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))
    dfour = np.fft.fftn(D)
    dksmo = np.zeros((N_dim, N_dim, N_dim),dtype=complex)
    ksq = np.zeros((N_dim, N_dim, N_dim),dtype=complex)
    ksq[:,:,:] = karr[None,None,:]**2+karr[None,:,None]**2+karr[:,None,None]**2
    dksmo[:,:,:] = Wg(ksq,rsmo)*dfour
    drsmo = np.real(np.fft.ifftn(dksmo))
    return drsmo

def get_moments(D_smo):
    second_moment = np.mean(D_smo**2)
    third_moment = np.mean(D_smo**3)
    print("Third = ",third_moment)
    return second_moment, third_moment

box_cents = np.linspace(0,Lbox,N_dim+1)
box_cents = .5*(box_cents[1:]+box_cents[:-1])
all_cents = np.array(list(itertools.product(box_cents, repeat=3)))


N_D = 3
Second = np.zeros((len(Rs),N_D**3))
Third = np.zeros((len(Rs),N_D**3))
Second_sam = np.zeros((len(Rs),N_D**3))
Third_sam = np.zeros((len(Rs),N_D**3))
Second_rat = np.zeros((len(Rs),N_D**3))
Third_rat = np.zeros((len(Rs),N_D**3))
for i in range(len(Rs)):
    R = Rs[i]
    print("R = ",R)
    D_g_smo = gaussian_filter(D_g,R)
    D_g_sam_smo = gaussian_filter(D_g_sam,R)
    D_g_smo = D_g_smo.flatten()
    D_g_sam_smo = D_g_sam_smo.flatten()
    for i_x in range(N_D):
        for i_y in range(N_D):
            for i_z in range(N_D):
                D_g_jack = D_g_smo.copy()
                D_g_sam_jack = D_g_sam_smo.copy()
                xyz_jack = all_cents.copy()

                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_D

                bool_arr = np.prod((xyz == (xyz_jack/size).astype(int)),axis=1).astype(bool)
                xyz_jack[bool_arr] = np.array([0.,0.,0.])
                this_chunk = np.sum(xyz_jack,axis=1)!=0.
                D_g_jack = D_g_jack[this_chunk]
                D_g_sam_jack = D_g_sam_jack[this_chunk]

                sec, thi = get_moments(D_g_jack)
                sec_o, thi_o = get_moments(D_g_sam_jack)

                Second[i,i_x+N_D*i_y+N_D**2*i_z] = sec
                Second_sam[i,i_x+N_D*i_y+N_D**2*i_z] = sec_o
                Second_rat[i,i_x+N_D*i_y+N_D**2*i_z] = sec_o/sec
                Third[i,i_x+N_D*i_y+N_D**2*i_z] = thi
                Third_sam[i,i_x+N_D*i_y+N_D**2*i_z] = thi_o
                Third_rat[i,i_x+N_D*i_y+N_D**2*i_z] = thi_o/thi
            
Second_mean = np.mean(Second,axis=1)
Second_err = np.sqrt(N_D**3-1)*np.std(Second,axis=1)
Third_mean = np.mean(Third,axis=1)
Third_err = np.sqrt(N_D**3-1)*np.std(Third,axis=1)
Second_sam_mean = np.mean(Second_sam,axis=1)
Second_sam_err = np.sqrt(N_D**3-1)*np.std(Second_sam,axis=1)
Second_rat_mean = np.mean(Second_rat,axis=1)
Second_rat_err = np.sqrt(N_D**3-1)*np.std(Second_rat,axis=1)
Third_sam_mean = np.mean(Third_sam,axis=1)
Third_sam_err = np.sqrt(N_D**3-1)*np.std(Third_sam,axis=1)
Third_rat_mean = np.mean(Third_rat,axis=1)
Third_rat_err = np.sqrt(N_D**3-1)*np.std(Third_rat,axis=1)

np.save("data/rs.npy",Rs)
np.save("data/second_true_mean.npy",Second_mean)
np.save("data/second_true_err.npy",Second_err)
np.save("data/second_"+sam+"_mean.npy",Second_sam_mean)
np.save("data/second_"+sam+"_err.npy",Second_sam_err)
np.save("data/second_rat_"+sam+"_mean.npy",Second_rat_mean)
np.save("data/second_rat_"+sam+"_err.npy",Second_rat_err)
np.save("data/third_true_mean.npy",Third_mean)
np.save("data/third_true_err.npy",Third_err)
np.save("data/third_"+sam+"_mean.npy",Third_sam_mean)
np.save("data/third_"+sam+"_err.npy",Third_sam_err)
np.save("data/third_rat_"+sam+"_mean.npy",Third_rat_mean)
np.save("data/third_rat_"+sam+"_err.npy",Third_rat_err)
