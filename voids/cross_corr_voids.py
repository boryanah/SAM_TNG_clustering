import matplotlib.pyplot as plt
import numpy as np
import sys
import Corrfunc
from Corrfunc.theory.DD import DD
from  mpi4py import MPI

Lbox = 205.
proxy = "m200m"
opts = ["shuff","partial_env_cw","partial_s2r","partial_vani","partial_tot_pot"]
n_random = 35#5#35

#ind = int(sys.argv[1])
ind = MPI.COMM_WORLD.Get_rank()
opt = opts[ind]
type_corr = 'cross'#'auto-vv'#'cross'

#ext1 = "data_2dhod_peak"
ext1 = "data_2dhod_pos"
#ext2 = "data_2dhod_peak"
ext2 = "data_2dhod_pos"
gal_dir = "/home/boryanah/lars/LSSIllustrisTNG/Lensing/"
test_name = '-'.join(opt.split('_'));print(test_name)
xyz_dm = np.load(gal_dir+ext1+"/" +"true_gals.npy").astype(np.float)
xyz_opt_dm = np.load(gal_dir+ext2+"/" +proxy+"_"+opt+"_gals.npy").astype(np.float)


void = np.load("data_clean/clean_void_true.npy")
xyz_void = void[:,1:]

void_opt = np.load("data_clean/clean_void_"+proxy+"_"+opt+".npy")
xyz_void_opt = void_opt[:,1:]

# TESTING
n_top = 1000
i_void = (np.argsort(void[:,0])[::-1])[:n_top]
xyz_void = xyz_void[i_void]
i_void_opt = (np.argsort(void_opt[:,0])[::-1])[:n_top]
xyz_void_opt = xyz_void_opt[i_void_opt]


'''
# TESTING
ind1 = 1; ind2 = 2
plt.figure(1)
plt.scatter(xyz_void_opt[:,ind1],xyz_void_opt[:,ind2],color='b',s=1,alpha=1)
plt.scatter(xyz_opt_dm[:,ind1],xyz_opt_dm[:,ind2],color='r',s=0.5,alpha=0.8)
plt.axis('equal')

plt.figure(2)
plt.scatter(xyz_void[:,ind1],xyz_void[:,ind2],color='b',s=1,alpha=1)
plt.scatter(xyz_dm[:,ind1],xyz_dm[:,ind2],color='r',s=0.5,alpha=0.8)
plt.axis('equal')
plt.show()
'''


# TESTING
#def get_cross(pos1,pos1_r,pos2,pos2_r,n_thread=16,periodic=True):
def get_cross(pos1,pos1_r,pos2,pos2_r,n1,n2,n_thread=16,periodic=True):
    X_jack_g = pos1[:,0]
    Y_jack_g = pos1[:,1]
    Z_jack_g = pos1[:,2]

    X_jack_m = pos2[:,0]
    Y_jack_m = pos2[:,1]
    Z_jack_m = pos2[:,2]

    # TESTING
    N_g = n1
    N_m = n2
    #N_g = len(X_jack_g)
    #N_m = len(X_jack_m)


    X_jack_r_m = pos2_r[:,0]
    Y_jack_r_m = pos2_r[:,1]
    Z_jack_r_m = pos2_r[:,2]
    
    X_jack_r_g = pos1_r[:,0]
    Y_jack_r_g = pos1_r[:,1]
    Z_jack_r_g = pos1_r[:,2]

    # POSSIBLE BUG IN LENSING
    # TESTING
    #N_r_g = len(X_jack_r_g)
    #N_r_m = len(X_jack_r_m)
    N_r_g = N_g*n_random
    N_r_m = N_m*n_random
    
    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_g, Y1=Y_jack_g, Z1=Z_jack_g,
                 X2=X_jack_m, Y2=Y_jack_m, Z2=Z_jack_m,
                 boxsize=Lbox,periodic=periodic)

    DD_gm = results['npairs'].astype(float)
    DD_gm /= (N_g*1.*N_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_g, Y1=Y_jack_g, Z1=Z_jack_g,
                 X2=X_jack_r_m, Y2=Y_jack_r_m, Z2=Z_jack_r_m,
                 boxsize=Lbox,periodic=periodic)


    DR_gm = results['npairs'].astype(float)
    DR_gm /= (N_g*1.*N_r_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_r_g, Y1=Y_jack_r_g, Z1=Z_jack_r_g,
                 X2=X_jack_m, Y2=Y_jack_m, Z2=Z_jack_m,
                 boxsize=Lbox,periodic=periodic)

    RD_gm = results['npairs'].astype(float)
    RD_gm /= (N_r_g*1.*N_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_r_g, Y1=Y_jack_r_g, Z1=Z_jack_r_g,
                 X2=X_jack_r_m, Y2=Y_jack_r_m, Z2=Z_jack_r_m,
                 boxsize=Lbox,periodic=periodic)


    RR_gm = results['npairs'].astype(float)
    RR_gm /= (N_r_g*1.*N_r_m)

    Corr_gm = (DD_gm-DR_gm-RD_gm+RR_gm)/RR_gm

    return Corr_gm

def get_pos(pos_g,xyz,size):
    pos_g_jack = pos_g.copy()
    bool_arr = np.prod((xyz == (pos_g/size).astype(int)),axis=1).astype(bool)
    pos_g_jack[bool_arr] = np.array([0.,0.,0.])
    pos_g_jack = pos_g_jack[np.sum(pos_g_jack,axis=1)!=0.]
    return pos_g_jack

def get_random(pos,dtype=np.float64):
    N = pos.shape[0]
    N_r = N*n_random
    pos_r = np.random.uniform(0.,Lbox,(N_r,3)).astype(dtype)
    return pos_r
    
def get_jack(pos1,pos2,pos3,pos4):
    pos1_r = get_random(pos1,pos1.dtype)
    pos2_r = get_random(pos2,pos2.dtype)
    pos3_r = get_random(pos3,pos3.dtype)
    pos4_r = get_random(pos4,pos4.dtype)

    # TESTING
    N1 = pos1.shape[0]
    N2 = pos2.shape[0]
    N3 = pos3.shape[0]
    N4 = pos4.shape[0]
        
    
    N_dim = 3
    size = Lbox/N_dim
    corr12 = np.zeros((N_bin-1,N_dim**3))
    corr34 = np.zeros((N_bin-1,N_dim**3))
    rat = np.zeros((N_bin-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                print(xyz)                
                xyz1_jack = get_pos(pos1,xyz,size)
                xyz2_jack = get_pos(pos2,xyz,size)
                xyz_r1_jack = get_pos(pos1_r,xyz,size)
                xyz_r2_jack = get_pos(pos2_r,xyz,size)

                xyz3_jack = get_pos(pos3,xyz,size)
                xyz4_jack = get_pos(pos4,xyz,size)
                xyz_r3_jack = get_pos(pos3_r,xyz,size)
                xyz_r4_jack = get_pos(pos4_r,xyz,size)

                # TESTING
                Corr12 = get_cross(xyz1_jack,xyz_r1_jack,xyz2_jack,xyz_r2_jack,n1=N1,n2=N2)
                #Corr12 = get_cross(xyz1_jack,xyz_r1_jack,xyz2_jack,xyz_r2_jack)
                corr12[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr12
                # TESTING
                #Corr34 = get_cross(xyz3_jack,xyz_r3_jack,xyz4_jack,xyz_r4_jack)
                Corr34 = get_cross(xyz3_jack,xyz_r3_jack,xyz4_jack,xyz_r4_jack,n1=N3,n2=N4)
                corr34[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr34

                rat[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr34/Corr12
                
    Corr12_mean = np.mean(corr12,axis=1)
    Corr12_error = np.sqrt(N_dim**3-1)*np.std(corr12,axis=1)
    Corr34_mean = np.mean(corr34,axis=1)
    Corr34_error = np.sqrt(N_dim**3-1)*np.std(corr34,axis=1)
    Rat_mean = np.mean(rat,axis=1)
    Rat_error = np.sqrt(N_dim**3-1)*np.std(rat,axis=1)
    return Corr12_mean, Corr12_error, Corr34_mean, Corr34_error, Rat_mean, Rat_error

N_bin = 9
bins = np.logspace(np.log10(0.8),np.log10(20),N_bin)
bin_centers = (bins[:-1] + bins[1:])/2.
power = 0

if type_corr == 'cross':
    cross_mean, cross_err, cross_opt_mean, cross_opt_err, rat_opt_mean, rat_opt_err = get_jack(xyz_void,xyz_dm,xyz_void_opt,xyz_opt_dm)

    # TESTING
    pos1, pos2, pos3, pos4 = xyz_void,xyz_dm,xyz_void_opt,xyz_opt_dm
    pos1_r = get_random(pos1,pos1.dtype)
    pos2_r = get_random(pos2,pos2.dtype)
    pos3_r = get_random(pos3,pos3.dtype)
    pos4_r = get_random(pos4,pos4.dtype)
    corr12 = get_cross(pos1,pos1_r,pos2,pos2_r,n1=pos1.shape[0],n2=pos2.shape[0])
    corr34 = get_cross(pos3,pos3_r,pos4,pos4_r,n1=pos3.shape[0],n2=pos4.shape[0])
    plt.plot(bin_centers, corr12*bin_centers**power,'b--',linewidth=2.,label='true')
    plt.plot(bin_centers, corr34*bin_centers**power,'r--',linewidth=2.,label='opt')
    
elif type_corr == 'auto-gg':
    cross_mean, cross_err, cross_opt_mean, cross_opt_err, rat_opt_mean, rat_opt_err = get_jack(xyz_dm,xyz_dm,xyz_opt_dm,xyz_opt_dm)

    # TESTING
    res_dm = Corrfunc.theory.xi(X=xyz_dm[:,0],Y=xyz_dm[:,1],Z=xyz_dm[:,2],boxsize=Lbox,nthreads=16,binfile=bins)
    res_opt_dm = Corrfunc.theory.xi(X=xyz_opt_dm[:,0],Y=xyz_opt_dm[:,1],Z=xyz_opt_dm[:,2],boxsize=Lbox,nthreads=16,binfile=bins)

    plt.plot(bin_centers, res_dm['xi']*bin_centers**power,'b--',linewidth=2.,label='true')
    plt.plot(bin_centers, res_opt_dm['xi']*bin_centers**power,'r--',linewidth=2.,label='opt')

elif type_corr == 'auto-vv':
    cross_mean, cross_err, cross_opt_mean, cross_opt_err, rat_opt_mean, rat_opt_err = get_jack(xyz_void,xyz_void,xyz_void_opt,xyz_void_opt)

    # TESTING
    res_dm = Corrfunc.theory.xi(X=xyz_void[:,0],Y=xyz_void[:,1],Z=xyz_void[:,2],boxsize=Lbox,nthreads=16,binfile=bins)
    res_opt_dm = Corrfunc.theory.xi(X=xyz_void_opt[:,0],Y=xyz_void_opt[:,1],Z=xyz_void_opt[:,2],boxsize=Lbox,nthreads=16,binfile=bins)

    plt.plot(bin_centers, res_dm['xi']*bin_centers**power,'b--',linewidth=2.,label='true')
    plt.plot(bin_centers, res_opt_dm['xi']*bin_centers**power,'r--',linewidth=2.,label='opt')


# TESTING
'''
plt.errorbar(bin_centers, cross_mean*bin_centers**power,yerr=cross_err*bin_centers**power,color='b',linewidth=2.,label='true')
plt.errorbar(bin_centers, cross_opt_mean*bin_centers**power,yerr=cross_opt_err*bin_centers**power,color='r',linewidth=2.,label='opt')
plt.xscale('log')
plt.ylim([-2,2])
plt.legend()
plt.savefig("cross_"+opt+".png")
plt.show()
'''

np.save("data_cross/bin_cents.npy",bin_centers)
np.save("data_cross/rat_cross_"+opt+"_mean.npy",rat_opt_mean)
np.save("data_cross/rat_cross_"+opt+"_error.npy",rat_opt_err)
np.save("data_cross/cross_true_mean.npy",cross_mean)
np.save("data_cross/cross_true_error.npy",cross_err)
np.save("data_cross/cross_"+opt+"_mean.npy",cross_opt_mean)
np.save("data_cross/cross_"+opt+"_error.npy",cross_opt_err)
quit()
#plt.errorbar(bin_centers, cross_mean*bin_centers**2,yerr=cross_err*bin_centers**2,color='b',linewidth=2.,label='true')
#plt.errorbar(bin_centers, cross_opt_mean*bin_centers**2,yerr=cross_opt_err*bin_centers**2,color='r',linewidth=2.,label='opt')
plt.plot(bin_centers,np.ones(len(bin_centers)),'k--',alpha=0.3)
#plt.plot(bin_centers, res_opt_dm/res_dm,'b',linewidth=2.,label='matched positions')
plt.errorbar(bin_centers,rat_opt_mean,yerr=rat_opt_err,color='r',linewidth=2.,label=opt+'/true')
plt.xscale('log')
plt.ylim([-3,4])
plt.legend()
plt.savefig("cross_"+opt+".png")
#plt.show()
