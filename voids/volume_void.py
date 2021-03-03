import matplotlib.pyplot as plt
import numpy as np
import sys
import itertools

proxy = "m200m"
#opts = ["spin","partial_fenv","partial_s2r","shuff","env_mass","partial_vani","partial_tot_pot","partial_min_pot"]
#opts = ["shuff","spin","partial_min_pot","partial_tot_pot","partial_fenv","partial_vani","partial_env_cw","partial_s2r","env_cw","tot_pot","min_pot","env_mass"]
#opts = ["tot_pot","env","partial_env_cw","partial_s2r","shuff","env_mass","partial_vani","partial_tot_pot"]

# For paper
opts = ["shuff","partial_env_cw","partial_vani","partial_tot_pot","partial_s2r"]

def estimate_volume(voids):
    void_g = voids[:,0]
    void_xyz = voids[:,1:]

    voids_inside = np.zeros(all_cents.shape[0])
    for j in range(len(void_g)):
        if j%100 == 0: print(j,len(void_g))
        v_s = void_g[j]
        v_p = void_xyz[j]
        d2 = np.sum((all_cents-v_p)**2,axis=1)
        voids_inside[np.sqrt(d2) < v_s] = 1.
        
    print("Takes up this fraction of the simulation = ",np.sum(voids_inside)/len(voids_inside))


void = np.load("data_clean/clean_void_true.npy")


N_r = len(opts)
Lbox = 205.
N_dim = 128#256
box_cents = np.linspace(0,Lbox,N_dim+1)
box_cents = .5*(box_cents[1:]+box_cents[:-1])
all_cents = np.array(list(itertools.product(box_cents, repeat=3)))

#estimate_volume(void);
#quit()

for i in range(N_r):
    i = 1
    opt = opts[i]
    opt_name = '-'.join(opt.split('_'));print(opt_name)
    print(opt)

    void_opt = np.load("data_clean/clean_void_"+proxy+"_"+opt+".npy")
    estimate_volume(void_opt)
    quit()
