import numpy as np
import matplotlib.pyplot as plt

import Corrfunc

#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Times'],'size':14})
#rc('text', usetex=True)

import plotparams
plotparams.buba()


# what property
type_prop = 'conc'
#type_prop = 'env'

if type_prop == 'conc':
    lab_prop = 'concentration'
elif type_prop == 'env':
    lab_prop = 'environment'
    
# size of box
Lbox = 205.

# mass bins
m_min = 12
m_max = 14.5

# save the low and high inds
save_inds = 0#True

#m_bins = np.logspace(m_min, m_max, 5)
#m_binc = (m_bins[1:] + m_bins[:-1])*.5
m_binc = np.linspace(12, 14.5, 6)
if save_inds:
    delta = 0.05 # save
else:
    delta = 0.3 #og

# how are we dividing into top and bottom
if save_inds:
    percentile = 5. # save
else:
    percentile = 30. # og

# sim params
sam_dir = '/mnt/alan1/boryanah/SAM_subvolumes_TNG300/'
hydro_dir = '/mnt/gosling1/boryanah/TNG300/'
str_snap = ''

# colors for plotting
color_sam = 'dodgerblue'
color_tng = '#CC6677'

# loading SAM
sat_type = np.load(sam_dir+'GalpropSatType'+str_snap+'.npy')
halo_pos = np.load(sam_dir+'GalpropPos'+str_snap+'.npy')[sat_type == 0]
halo_mvir = np.load(sam_dir+'GalpropMhalo'+str_snap+'.npy')[sat_type == 0]
if type_prop == 'env':
    halo_env = np.load(sam_dir+'HalopropEnvironment'+str_snap+'.npy')
elif type_prop == 'conc':
    halo_env = np.load(sam_dir+'HalopropC_nfw'+str_snap+'.npy')


# sort all arrays in order of halo mass
i_sort = np.argsort(halo_mvir)[::-1]
halo_mvir = halo_mvir[i_sort]
halo_pos = halo_pos[i_sort]
halo_env = halo_env[i_sort]

# correct positions
halo_pos[halo_pos < 0] += Lbox
halo_pos[halo_pos >= Lbox] -= Lbox

# loading TNG
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp'+str_snap+'.npy')/1000.
#GrMcrit_fp = np.load(hydro_dir+'Group_M_TopHat200_fp'+str_snap+'.npy')*1.e10
GrMcrit_fp = np.load(hydro_dir+'GroupMassType_fp'+str_snap+'.npy')[:, 1]*1.e10
if type_prop == 'env':
    GroupEnv_fp = np.load(hydro_dir+'GroupEnv_fp'+str_snap+'.npy')
elif type_prop == 'conc':
    GroupEnv_fp = np.load(hydro_dir+'GroupConc_nfw_fp'+str_snap+'.npy')
    #GroupEnv_fp = np.load(hydro_dir+'GroupConc_fp'+str_snap+'.npy')


# check savings
inds_high = np.load("visuals_fof/inds_high.npy")
print("high masses:")
for ind_high in inds_high:
    print(f'{GrMcrit_fp[ind_high]:.3e}', f'{GroupEnv_fp[ind_high]:.1f}')

inds_low = np.load("visuals_fof/inds_low.npy")
print("low masses:")
for ind_low in inds_low:
    print(f'{GrMcrit_fp[ind_low]:.3e}', f'{GroupEnv_fp[ind_low]:.1f}')



# sort all arrays in order of halo mass
i_sort = np.argsort(GrMcrit_fp)[::-1]
i_sort_rev = np.argsort(i_sort)
GrMcrit_fp = GrMcrit_fp[i_sort]
GroupPos_fp = GroupPos_fp[i_sort]
GroupEnv_fp = GroupEnv_fp[i_sort]

# correct positions
GroupPos_fp[GroupPos_fp < 0] += Lbox
GroupPos_fp[GroupPos_fp >= Lbox] -= Lbox

# corrfunc specs
bins = np.logspace(-0.7, 1.5, 11)
bin_cents = (bins[1:] + bins[:-1])*.5

# if we have fewer than thresh, bad things
if save_inds:
    thresh = 20
else:
    thresh = 2000 #og


nrow = 2
ncol = len(m_binc)//nrow
plt.subplots(nrow, ncol, figsize=(17, 10))
for i in range(len(m_binc)):
    # mass bin
    #m_thr1 = m_bins[i]
    #m_thr2 = m_bins[i+1]
    m_thr1 = 10.**(m_binc[i]-delta)
    m_thr2 = 10.**(m_binc[i]+delta)
    
    # choose halos above mass threshold
    group_choice = (GrMcrit_fp > m_thr1) & (GrMcrit_fp < m_thr2)
    n_choice = np.sum(group_choice)
    inds_choice = np.arange(len(GrMcrit_fp), dtype=int)[group_choice]
    print("number of halos in mass bin = ", n_choice)
    if n_choice < thresh:
        ind_thr1 = inds_choice[0] # smaller number
        #ind_thr2 = inds_choice[-1] # larger number
        ind_thr2 = ind_thr1 + thresh
        inds_choice = np.arange(ind_thr1, ind_thr2)
    group_choice = inds_choice
    halo_choice = inds_choice

    
    # corresponding indices for other finder
    #halo_choice = (halo_mvir > m_thr1) & (halo_mvir < m_thr2)
    
    print("number of halos for both = ", len(halo_choice), len(group_choice))
    
    # select the densities in that mass bin
    group_envi = GroupEnv_fp[group_choice]
    halo_envi = halo_env[halo_choice]

    # select the positions in that mass bin
    halo_posi = halo_pos[halo_choice]
    group_posi = GroupPos_fp[group_choice]

    # HIGH THRESHOLD
    
    # find the high thresholds
    group_env_thr = np.percentile(group_envi, 100-percentile)
    halo_env_thr = np.percentile(halo_envi, 100-percentile)

    # find the halo positions
    halo_p = halo_posi[halo_env_thr < halo_envi]
    group_p = group_posi[group_env_thr < group_envi]

    if save_inds:
        if i == 3:
            print("logM = ", m_binc[i])
            inds_high = group_choice[group_env_thr < group_envi]
            inds_high = i_sort[inds_high]
            envs = (GroupEnv_fp[i_sort_rev])[inds_high]
            #envs = (GroupEnv_fp)[inds_high]
            print("high environments = ", envs)
            print("lowest = ", np.min(envs))
        
    print("sum high = ", np.sum(halo_env_thr < halo_envi), np.sum(group_env_thr < group_envi))

    # parse x, y, z
    halo_x = halo_p[:, 0]; halo_y = halo_p[:, 1]; halo_z = halo_p[:, 2]
    group_x = group_p[:, 0]; group_y = group_p[:, 1]; group_z = group_p[:, 2]

    # compute corrfunc
    halo_xi_hi = Corrfunc.theory.xi(X=halo_x, Y=halo_y, Z=halo_z, boxsize=Lbox, nthreads=16, binfile=bins)['xi']
    group_xi_hi = Corrfunc.theory.xi(X=group_x, Y=group_y, Z=group_z, boxsize=Lbox, nthreads=16, binfile=bins)['xi']

    # MEDIAN THRESHOLD
    
    # find the median thresholds
    halo_env_thr1 = np.percentile(halo_envi, 50.-percentile/2.)
    halo_env_thr2 = np.percentile(halo_envi, 50.+percentile/2.)
    group_env_thr1 = np.percentile(group_envi, 50.-percentile/2.)
    group_env_thr2 = np.percentile(group_envi, 50.+percentile/2.)

    # find the halo positions
    halo_p = halo_posi[(halo_envi > halo_env_thr1) & (halo_envi < halo_env_thr2)]
    group_p = group_posi[(group_envi > group_env_thr1) & (group_envi < group_env_thr2)]

    # parse x, y, z
    halo_x = halo_p[:, 0]; halo_y = halo_p[:, 1]; halo_z = halo_p[:, 2]
    group_x = group_p[:, 0]; group_y = group_p[:, 1]; group_z = group_p[:, 2]

    # compute corrfunc
    halo_xi = Corrfunc.theory.xi(X=halo_x, Y=halo_y, Z=halo_z, boxsize=Lbox, nthreads=16, binfile=bins)['xi']
    group_xi = Corrfunc.theory.xi(X=group_x, Y=group_y, Z=group_z, boxsize=Lbox, nthreads=16, binfile=bins)['xi']


    # LOW THRESHOLD

    # find the low thresholds
    group_env_thr = np.percentile(group_envi, percentile)
    halo_env_thr = np.percentile(halo_envi, percentile)

    # find the halo positions
    halo_p = halo_posi[halo_env_thr > halo_envi]
    group_p = group_posi[group_env_thr > group_envi]

    if save_inds:
        if i == 3:
            print("logM = ", m_binc[i])
            inds_low = group_choice[group_env_thr > group_envi]
            #envs = (GroupEnv_fp)[inds_low]
            inds_low = i_sort[inds_low]
            envs = (GroupEnv_fp[i_sort_rev])[inds_low]
            print("low environments = ", envs)
            print("highest = ", np.max(envs))
    print("sum low = ", np.sum(halo_env_thr > halo_envi), np.sum(group_env_thr > group_envi))

    # parse x, y, z
    halo_x = halo_p[:, 0]; halo_y = halo_p[:, 1]; halo_z = halo_p[:, 2]
    group_x = group_p[:, 0]; group_y = group_p[:, 1]; group_z = group_p[:, 2]

    # compute corrfunc
    halo_xi_lo = Corrfunc.theory.xi(X=halo_x, Y=halo_y, Z=halo_z, boxsize=Lbox, nthreads=16, binfile=bins)['xi']
    group_xi_lo = Corrfunc.theory.xi(X=group_x, Y=group_y, Z=group_z, boxsize=Lbox, nthreads=16, binfile=bins)['xi']
    
    # show result
    plt.subplot(nrow, ncol, i+1)
    plt.plot(bin_cents, np.ones(len(bin_cents)), 'k--')
    
    plt.plot(bin_cents, group_xi_hi/group_xi, color=color_tng, label=f'TNG, high {lab_prop:s}')#top %d'%percentile)
    plt.plot(bin_cents, halo_xi_hi/halo_xi, color=color_sam, label=f'SAM, high {lab_prop:s}')#top %d'%percentile)
    plt.plot(bin_cents, group_xi_lo/group_xi, color=color_tng, ls='--', label=f'TNG, low {lab_prop:s}')#bottom %d'%percentile)
    plt.plot(bin_cents, halo_xi_lo/halo_xi, color=color_sam, ls='--', label=f'SAM, low {lab_prop:s}')#bottom %d'%percentile)
    plt.xscale('log')
    if i == 0:
        plt.legend(loc='upper left', frameon=False, fontsize=14)

    plt.text(0.75, 0.9, r"$\log M = %.1f$"%m_binc[i], ha='center', va='center', transform=plt.gca().transAxes)
        
    if type_prop == 'conc':
        plt.ylim([0, 8.])
    elif type_prop == 'env':
        plt.ylim([0, 8.])
    plt.xlabel(r"$r \ {\rm Mpc}/h$")
    if i%ncol == 0:
        plt.ylabel(r"$\xi_{\rm high, low}(r) / \xi_{\rm median}(r)$")

if save_inds:
    np.save("visuals_fof/inds_high.npy", inds_high)
    np.save("visuals_fof/inds_low.npy", inds_low)
plt.savefig("figs/hab_"+type_prop+".png")
plt.show()
