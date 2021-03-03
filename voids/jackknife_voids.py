import matplotlib.pyplot as plt
import numpy as np
import sys

proxy = "m200m"
#opts = ["partial_env_cw","partial_s2r","shuff","partial_vani","partial_tot_pot"]

opts = ["norm"]

void = np.load("data/clean_void_true.npy")
void_g = void[:,0]
void_xyz = void[:,1:]

N_r = len(opts)
Lbox = 205.
N_dim = 3
N_bin = 21
for i in range(N_r):
    opt = opts[i]
    opt_name = '-'.join(opt.split('_'));print(opt_name)
    void_opt = np.load("data/clean_void_"+opt+".npy")
    void_g_opt = void_opt[:,0]
    void_xyz_opt = void_opt[:,1:]
    
    rat_g = np.zeros((N_bin-1,N_dim**3))
    hist_g = np.zeros((N_bin-1,N_dim**3))
    hist_g_opt = np.zeros((N_bin-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                # og
                #min_bin = 10.#np.min(np.hstack((void_g,void_g_opt)))
                # TESTING
                min_bin = 1.
                max_bin = 24.#np.max(np.hstack((void_g,void_g_opt)))
                bins = np.linspace(min_bin,max_bin,N_bin)
                #bins = np.logspace(np.log10(min_bin),np.log10(max_bin),N_bin)
                c = .5*(bins[1:]+bins[:-1])

                void_jack = void_g.copy()
                void_opt_jack = void_g_opt.copy()
                xyz_jack = void_xyz.copy()
                xyz_opt_jack = void_xyz_opt.copy()
            
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_jack/size).astype(int)),axis=1).astype(bool)
                xyz_jack[bool_arr] = np.array([0.,0.,0.])
                this_chunk = np.sum(xyz_jack,axis=1)!=0.
                void_jack = void_jack[this_chunk]

                bool_arr = np.prod((xyz == (xyz_opt_jack/size).astype(int)),axis=1).astype(bool)
                xyz_opt_jack[bool_arr] = np.array([0.,0.,0.])
                this_chunk = np.sum(xyz_opt_jack,axis=1)!=0.
                void_opt_jack = void_opt_jack[this_chunk]
                #xyz_jack = xyz_jack[this_chunk]

                h_g, ed_g = np.histogram(void_jack,bins=bins)
                h_g_opt, ed_g_opt = np.histogram(void_opt_jack,bins=bins)
                rat_g[:,i_x+N_dim*i_y+N_dim**2*i_z] = h_g_opt/h_g
                hist_g_opt[:,i_x+N_dim*i_y+N_dim**2*i_z] = h_g_opt
                hist_g[:,i_x+N_dim*i_y+N_dim**2*i_z] = h_g

    hist_g_mean = np.mean(hist_g,axis=1)
    hist_g_err = np.sqrt(N_dim**3-1)*np.std(hist_g,axis=1)

    hist_g_opt_mean = np.mean(hist_g_opt,axis=1)
    hist_g_opt_err = np.sqrt(N_dim**3-1)*np.std(hist_g_opt,axis=1)

    rat_g_mean = np.mean(rat_g,axis=1)
    rat_g_err = np.sqrt(N_dim**3-1)*np.std(rat_g,axis=1)

    np.save("data/bin_centers.npy",c)
    np.save("data/void_true_mean.npy",hist_g_mean)
    np.save("data/void_true_err.npy",hist_g_err)
    np.save("data/void_"+opt+"_mean.npy",hist_g_opt_mean)
    np.save("data/void_"+opt+"_err.npy",hist_g_opt_err)
    np.save("data/void_"+opt+"_rat_mean.npy",rat_g_mean)
    np.save("data/void_"+opt+"_rat_err.npy",rat_g_err)

