import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import Corrfunc
from Corrfunc.theory.DD import DD
import plotparams
plotparams.default()
import sys
from halotools.utils import randomly_downsample_data

plt.rc('text', usetex=True)

def get_RR_norm(bins,Lbox,N1=1,N2=1):
    vol_all = 4./3*np.pi*bins**3
    vol_bins = vol_all[1:]-vol_all[:-1]
    n2 = N2/Lbox**3
    n2_bin = vol_bins*n2
    pairs = N1*n2_bin
    pairs /= (N1*N2)
    return pairs
                            

n_thread = 16
periodic = True
dir_part = "/mnt/gosling1/boryanah/TNG300/"
#part_fn = "parts_position_tng300-3_99.npy"
part_fn = "pos_parts_tng300-2_99.npy"
if 'tng300-2' in part_fn: N_m = 1250**3
if 'tng300-3' in part_fn: N_m = 625**3

proxy = "m200m"
ext2 = "data_2dhod_peak" # place satellites on vpeaks
ext1 = "data_2dhod_pos" # this is currentyl only used for the true gals positions
opt = sys.argv[1]#"partial_fenv"#"partial_s2r"#"shuff"#"spin"#"shuff"#"conc"#"vani"#"shuff"#"env"#"partial_vani""partial_s2r""vdisp"

want_full = 1 # include all halos and don't exclude top 100 as for rest of paper
Lbox = 205. # in Mpc/h
pcle_mass = 3.03e9 #M_solar/h
# fiducial means are you looking at true galaxies or not
if len(sys.argv) > 2:
    fiducial = int(sys.argv[2])#True
else:
    fiducial = False
    

test_name = '-'.join(opt.split('_'));print(test_name)#proxy+"_"+opt

# load true gal pos and opt gal pos
# og
#pos_g = np.load("../"+ext1+"/true_gals.npy").astype(np.float32)
#pos_g = np.load("../"+ext2+"/true_gals.npy").astype(np.float32)

# TESTING with full
pos_g = np.load("../"+ext2+"/true_gals_all.npy").astype(np.float32)

# TESTING
'''
root = '/home/boryanah/lars/data_2500/'
SubhaloPos_fp = np.load(root+'SubhaloPos_fp.npy')/1000.
SubhaloMassType_fp = np.load(root+'SubhaloMassType_fp.npy')*1.e10
lum_proxy = SubhaloMassType_fp[:,4]
inds_top = (np.argsort(lum_proxy)[::-1])[869:11665]#pos_g.shape[0]]
pos_g = SubhaloPos_fp[inds_top]
'''
if want_full:
    pos_g_opt = np.load("../"+ext2+"/"+proxy+"_"+opt+"_gals_all.npy").astype(np.float32)
else:
    pos_g_opt = np.load("../"+ext2+"/"+proxy+"_"+opt+"_gals.npy").astype(np.float32)

if fiducial == False:
    pos_g = pos_g_opt.copy()

down = 50000# for tng300-2 # 5000 for tng300-3
N_m_down = N_m//down
print("Number of pcles after downsampling = ",N_m_down)
try:
    pos_m = np.load("../pos_m_down_"+str(down)+".npy")
    #pos_m = np.load("../pos_parts_down_610351.npy")
except:
    print("Downsampling...")
    pos_m = np.load(dir_part+part_fn)/1000. # in Mpc/h

    inds_m = np.arange(N_m)
    np.random.shuffle(inds_m)
    inds_m = inds_m[::down]
    pos_m = pos_m[inds_m]
    #x_m = pos_m[::down,0]
    #y_m = pos_m[::down,1]
    #z_m = pos_m[::down,2]
    #pos_m = np.vstack((x_m,y_m,z_m)).T

    #pos_m = randomly_downsample_data(pos_m, N_m_down)

    print(pos_m.shape[0])
    np.save("../pos_m_down_"+str(down)+".npy",pos_m)
    #plt.scatter(pos_m[:50000,0],pos_m[:50000,1],s=0.01)
    #plt.show()
print("Downsampled O.o!")

down_fac = N_m/N_m_down
pcle_mass *= down_fac

N_g = pos_g.shape[0]
N_g_opt = pos_g.shape[0]
print("Number of gals = ",N_g)

pos_m = pos_m.astype(np.float32)
X_m = pos_m[:,0]
Y_m = pos_m[:,1]
Z_m = pos_m[:,2]

X_g = pos_g[:,0]
Y_g = pos_g[:,1]
Z_g = pos_g[:,2]
    

N_bin = 16
N_dim = 3
lb_min = -0.7
lb_max = 1.2
bins = np.logspace(lb_min,lb_max,N_bin)
x_lim_min = 10**(lb_min)
x_lim_max = 10**(lb_max)
bin_centers = (bins[:-1] + bins[1:])/2.

xyz_m = np.vstack((X_m,Y_m,Z_m)).T
xyz_g = np.vstack((X_g,Y_g,Z_g)).T


testing = 1
if testing:
    N_g = len(X_g)
    N_m_down = len(X_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_g, Y1=Y_g, Z1=Z_g,
                 X2=X_m, Y2=Y_m, Z2=Z_m,
                 boxsize=Lbox,periodic=periodic)

    DD_gm = results['npairs'].astype(float)
    DD_gm /= (N_g*1.*N_m_down)
    RR_gm = get_RR_norm(bins,Lbox)

    Corr_gm = DD_gm/RR_gm-1.


    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_m, Y1=Y_m, Z1=Z_m,
                 X2=X_m, Y2=Y_m, Z2=Z_m,
                 boxsize=Lbox,periodic=periodic)

    DD_mm = results['npairs'].astype(float)
    DD_mm /= (N_m_down*1.*N_m_down)
    RR_mm = get_RR_norm(bins,Lbox)

    Corr_mm = DD_mm/RR_mm-1.

    Corr_gg = Corrfunc.theory.xi(X=X_g,Y=Y_g,Z=Z_g,boxsize=Lbox,nthreads=16,binfile=bins)['xi']
    '''
    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_g, Y1=Y_g, Z1=Z_g,
                 X2=X_g, Y2=Y_g, Z2=Z_g,
                 boxsize=Lbox,periodic=periodic)

    DD_gg = results['npairs'].astype(float)
    DD_gg /= (N_g*1.*N_g)
    RR_gg = get_RR_norm(bins,Lbox)
    Corr_gg = DD_gg/RR_gg-1.
    '''


    bias = np.sqrt(Corr_gg/Corr_mm)
    corr_coeff = Corr_gm/np.sqrt(Corr_gg*Corr_mm) 

    plt.figure(1)
    plt.plot(bin_centers,bias)
    plt.xscale('log')

    plt.figure(2)
    plt.plot(bin_centers,corr_coeff)
    plt.xscale('log')

    plt.show()
    quit()


bias = np.zeros((N_bin-1,N_dim**3))
corr_coeff = np.zeros((N_bin-1,N_dim**3))
corr_g = np.zeros((N_bin-1,N_dim**3))
corr_m = np.zeros((N_bin-1,N_dim**3))
corr_gm = np.zeros((N_bin-1,N_dim**3))
# JACKKNIFE ERROR ESTIMATION
for i_x in range(N_dim):
    for i_y in range(N_dim):
        for i_z in range(N_dim):
            xyz_g_jack = xyz_g.copy()
            xyz_m_jack = xyz_m.copy()
            
            
            xyz = np.array([i_x,i_y,i_z],dtype=int)
            size = Lbox/N_dim

            bool_arr = np.prod((xyz == (xyz_g/size).astype(int)),axis=1).astype(bool)
            xyz_g_jack[bool_arr] = np.array([0.,0.,0.])
            xyz_g_jack = xyz_g_jack[np.sum(xyz_g_jack,axis=1)!=0.]

            bool_arr = np.prod((xyz == (xyz_m/size).astype(int)),axis=1).astype(bool)
            xyz_m_jack[bool_arr] = np.array([0.,0.,0.])
            xyz_m_jack = xyz_m_jack[np.sum(xyz_m_jack,axis=1)!=0.]
            
            X_jack_m = xyz_m_jack[:,0]
            Y_jack_m = xyz_m_jack[:,1]
            Z_jack_m = xyz_m_jack[:,2]
            
            X_jack_g = xyz_g_jack[:,0]
            Y_jack_g = xyz_g_jack[:,1]
            Z_jack_g = xyz_g_jack[:,2]

            N_g = len(X_jack_g)
            N_m_down = len(X_jack_m)
            # Cross-correlation for gm
            print("Nightmare is starting")
            autocorr = 0
            results = DD(autocorr,nthreads=n_thread,binfile=bins,
                         X1=X_jack_g, Y1=Y_jack_g, Z1=Z_jack_g,
                         X2=X_jack_m, Y2=Y_jack_m, Z2=Z_jack_m,
                         boxsize=Lbox,periodic=periodic)

            DD_gm = results['npairs'].astype(float)
            DD_gm /= (N_g*1.*N_m_down)
            RR_gm = get_RR_norm(bins,Lbox)

            Corr_gm = DD_gm/RR_gm-1.

            # Corr_g
            #Corr_g = Corrfunc.theory.xi(X=X_jack_g,Y=Y_jack_g,Z=Z_jack_g,boxsize=Lbox,nthreads=16,binfile=bins)['xi']
            
            autocorr = 1
            results = DD(autocorr,nthreads=n_thread,binfile=bins,
                         X1=X_jack_g, Y1=Y_jack_g, Z1=Z_jack_g,
                         boxsize=Lbox,periodic=periodic)

            DD_gg = results['npairs'].astype(float)
            DD_gg /= (N_g*1.*N_g)
            RR_gg = get_RR_norm(bins,Lbox)
            
            Corr_g = DD_gg/RR_gg-1.
            
            
            # Corr_m
            #Corr_m = Corrfunc.theory.xi(X=X_jack_m,Y=Y_jack_m,Z=Z_jack_m,boxsize=Lbox,nthreads=16,binfile=bins)['xi']
            
            autocorr = 1
            results = DD(autocorr,nthreads=n_thread,binfile=bins,
                         X1=X_jack_m, Y1=Y_jack_m, Z1=Z_jack_m,
                         boxsize=Lbox,periodic=periodic)

            DD_mm = results['npairs'].astype(float)
            DD_mm /= (N_m_down*1.*N_m_down)
            RR_mm = get_RR_norm(bins,Lbox)
            
            Corr_m = DD_mm/RR_mm-1.
            
            
            bias[:,i_x+N_dim*i_y+N_dim**2*i_z] = np.sqrt(Corr_g/Corr_m)
            corr_coeff[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr_gm/np.sqrt(Corr_g*Corr_m) 
            corr_g[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr_g
            corr_m[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr_m
            corr_gm[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr_gm


bias_mean = np.mean(bias,axis=1)
bias_error = np.sqrt(N_dim**3-1)*np.std(bias,axis=1)
corr_coeff_mean = np.mean(corr_coeff,axis=1)
corr_coeff_error = np.sqrt(N_dim**3-1)*np.std(corr_coeff,axis=1)
Corr_gm_mean = np.mean(corr_gm,axis=1)
Corr_gm_error = np.sqrt(N_dim**3-1)*np.std(corr_gm,axis=1)
Corr_g_mean = np.mean(corr_g,axis=1)
Corr_g_error = np.sqrt(N_dim**3-1)*np.std(corr_g,axis=1)
Corr_m_mean = np.mean(corr_m,axis=1)
Corr_m_error = np.sqrt(N_dim**3-1)*np.std(corr_m,axis=1)


if fiducial:
    np.save("../data_gm/bin_fid.npy",bin_centers)
    np.save("../data_gm/bias_mean_fid.npy",bias_mean)
    np.save("../data_gm/corr_coeff_mean_fid.npy",corr_coeff_mean)
    np.save("../data_gm/corr_gm_mean_fid.npy",Corr_gm_mean)
    np.save("../data_gm/corr_gg_mean_fid.npy",Corr_g_mean)
    np.save("../data_gm/bias_error_fid.npy",bias_error)
    np.save("../data_gm/corr_coeff_error_fid.npy",corr_coeff_error)
    np.save("../data_gm/corr_gm_error_fid.npy",Corr_gm_error)
    np.save("../data_gm/corr_gg_error_fid.npy",Corr_g_error)
else:
    np.save("../data_gm/bin_cents.npy",bin_centers)
    np.save("../data_gm/bias_mean_"+opt+".npy",bias_mean)
    np.save("../data_gm/corr_coeff_mean_"+opt+".npy",corr_coeff_mean)
    np.save("../data_gm/corr_gm_mean_"+opt+".npy",Corr_gm_mean)
    np.save("../data_gm/corr_gg_mean_"+opt+".npy",Corr_g_mean)
    np.save("../data_gm/bias_error_"+opt+".npy",bias_error)
    np.save("../data_gm/corr_coeff_error_"+opt+".npy",corr_coeff_error)
    np.save("../data_gm/corr_gm_error_"+opt+".npy",Corr_gm_error)
    np.save("../data_gm/corr_gg_error_"+opt+".npy",Corr_g_error)
    
