import numpy as np
import scipy.spatial as spatial
import itertools
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# do the computation for the true galaxies?
want_true = 0

# what are we displaying
proxy = "m200m"
opt = sys.argv[1]#"partial_fenv"#"partial_s2r"#"shuff"#"spin"#"shuff"#"conc"#"vani"#"shuff"#"env"#"partial_vani""partial_s2r""vdisp"

# parameter choices
Lbox = 205. # in Mpc/h
bounds = np.array([Lbox,Lbox,Lbox])
N_dim = 256
gr_size = Lbox/N_dim
Rs = np.linspace(3,8,6)

# load them galaxies
ext2 = "data_2dhod_peak"
ext1 = "data_2dhod_pos"
gal_dir = "/home/boryanah/lars/LSSIllustrisTNG/Lensing/"
test_name = '-'.join(opt.split('_'));print(test_name)
pos_g = np.load(gal_dir+ext1+"/"+"true_gals.npy")
pos_g_opt = np.load(gal_dir+ext2+"/"+proxy+"_"+opt+"_gals.npy")
    
# how mangy galaxies
N_g = pos_g.shape[0]
N_g_opt = pos_g_opt.shape[0]
print("Number of gals = ",N_g_opt)

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
D_g_opt = get_density(pos_g_opt)


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
Second_opt = np.zeros((len(Rs),N_D**3))
Third_opt = np.zeros((len(Rs),N_D**3))
Second_rat = np.zeros((len(Rs),N_D**3))
Third_rat = np.zeros((len(Rs),N_D**3))
for i in range(len(Rs)):
    R = Rs[i]
    print("R = ",R)
    D_g_smo = gaussian_filter(D_g,R)
    D_g_opt_smo = gaussian_filter(D_g_opt,R)
    D_g_smo = D_g_smo.flatten()
    D_g_opt_smo = D_g_opt_smo.flatten()
    for i_x in range(N_D):
        for i_y in range(N_D):
            for i_z in range(N_D):
                D_g_jack = D_g_smo.copy()
                D_g_opt_jack = D_g_opt_smo.copy()
                xyz_jack = all_cents.copy()

                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_D

                bool_arr = np.prod((xyz == (xyz_jack/size).astype(int)),axis=1).astype(bool)
                xyz_jack[bool_arr] = np.array([0.,0.,0.])
                this_chunk = np.sum(xyz_jack,axis=1)!=0.
                D_g_jack = D_g_jack[this_chunk]
                D_g_opt_jack = D_g_opt_jack[this_chunk]

                sec, thi = get_moments(D_g_jack)
                sec_o, thi_o = get_moments(D_g_opt_jack)

                Second[i,i_x+N_D*i_y+N_D**2*i_z] = sec
                Second_opt[i,i_x+N_D*i_y+N_D**2*i_z] = sec_o
                Second_rat[i,i_x+N_D*i_y+N_D**2*i_z] = sec_o/sec
                Third[i,i_x+N_D*i_y+N_D**2*i_z] = thi
                Third_opt[i,i_x+N_D*i_y+N_D**2*i_z] = thi_o
                Third_rat[i,i_x+N_D*i_y+N_D**2*i_z] = thi_o/thi
            
Second_mean = np.mean(Second,axis=1)
Second_err = np.sqrt(N_D**3-1)*np.std(Second,axis=1)
Third_mean = np.mean(Third,axis=1)
Third_err = np.sqrt(N_D**3-1)*np.std(Third,axis=1)
Second_opt_mean = np.mean(Second_opt,axis=1)
Second_opt_err = np.sqrt(N_D**3-1)*np.std(Second_opt,axis=1)
Second_rat_mean = np.mean(Second_rat,axis=1)
Second_rat_err = np.sqrt(N_D**3-1)*np.std(Second_rat,axis=1)
Third_opt_mean = np.mean(Third_opt,axis=1)
Third_opt_err = np.sqrt(N_D**3-1)*np.std(Third_opt,axis=1)
Third_rat_mean = np.mean(Third_rat,axis=1)
Third_rat_err = np.sqrt(N_D**3-1)*np.std(Third_rat,axis=1)

np.save("data_many/rs.npy",Rs)
np.save("data_many/second_true_mean.npy",Second_mean)
np.save("data_many/second_true_err.npy",Second_err)
np.save("data_many/second_"+proxy+"_"+opt+"_mean.npy",Second_opt_mean)
np.save("data_many/second_"+proxy+"_"+opt+"_err.npy",Second_opt_err)
np.save("data_many/second_rat_"+proxy+"_"+opt+"_mean.npy",Second_rat_mean)
np.save("data_many/second_rat_"+proxy+"_"+opt+"_err.npy",Second_rat_err)
np.save("data_many/third_true_mean.npy",Third_mean)
np.save("data_many/third_true_err.npy",Third_err)
np.save("data_many/third_"+proxy+"_"+opt+"_mean.npy",Third_opt_mean)
np.save("data_many/third_"+proxy+"_"+opt+"_err.npy",Third_opt_err)
np.save("data_many/third_rat_"+proxy+"_"+opt+"_mean.npy",Third_rat_mean)
np.save("data_many/third_rat_"+proxy+"_"+opt+"_err.npy",Third_rat_err)
