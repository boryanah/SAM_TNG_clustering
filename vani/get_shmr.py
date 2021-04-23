import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# matplotlib settings
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times'],'size':18})
rc('text', usetex=True)
color1_sam = '#CC6677'
color1_tng = 'dodgerblue'
color2_sam = 'crimson'
color2_tng = 'steelblue'
color3_sam = 'darkred'
color3_tng = 'navy'




# sim params
sam_dir = '/mnt/alan1/boryanah/SAM_subvolumes_TNG300/'
hydro_dir = '/mnt/gosling1/boryanah/TNG300/'
#str_snap = '_55'; zuni = "1.00"; h = 0.6774
str_snap = ''; zuni = "0.10"; h = 0.6774
Lbox = 205.
fac = 2
f_sat = 0.25
num_gals = 12000

# loading SAM
sat_type = np.load(sam_dir+'GalpropSatType'+str_snap+'.npy')
halo_mvir = np.load(sam_dir+'GalpropMhalo'+str_snap+'.npy')[sat_type == 0]
mstar = np.load(sam_dir+'GalpropMstar'+str_snap+'.npy')
mvir = np.load(sam_dir+'GalpropMhalo'+str_snap+'.npy')
halo_mstar = np.load(sam_dir+'GalpropMstar'+str_snap+'.npy')[sat_type == 0]
hosthaloid = np.load(sam_dir+'GalpropHaloIndex'+str_snap+'.npy').astype(int)

# loading TNG 
#SubhaloMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp'+str_snap+'.npy')[:, 4]*1.e10
#SubhaloMstar_fp = np.load(hydro_dir+'SubhaloMassInHalfRadType_fp'+str_snap+'.npy')[:, 4]*1.e10
SubhaloMstar_fp = np.load(hydro_dir+'SubhaloMassInRadType_fp'+str_snap+'.npy')[:, 4]*1.e10
#rescaled = 1.4
#SubhaloMstar_fp *= rescaled

SubhaloGrNr = np.load(hydro_dir+"SubhaloGrNr_fp"+str_snap+".npy")
#Group_M_Mean200 = np.load(hydro_dir+"Group_M_Mean200_fp"+str_snap+".npy")*1.e10
Group_M_Mean200 = np.load(hydro_dir+"Group_M_TopHat200_fp"+str_snap+".npy")*1.e10
#Group_M_Mean200 = np.load(hydro_dir+"Group_M_Crit200_fp"+str_snap+".npy")*1.e10

# abundance matching selection for the SAM move lower but note the sorting of mvir
inds_top = (np.argsort(mstar)[::-1])[:num_gals*fac]
inds_gal = np.arange(len(mstar), dtype=int)
inds_cent = inds_gal[sat_type == 0]
inds_sats = inds_gal[sat_type != 0]
bool_top_cent = np.in1d(inds_top, inds_cent)
bool_top_sats = np.in1d(inds_top, inds_sats)
num_sats = int(np.round(f_sat*num_gals))
num_cent = num_gals - num_sats
inds_top_cent = (inds_top[bool_top_cent])[:num_cent]
inds_top_sats = (inds_top[bool_top_sats])[:num_sats]
inds_top = np.hstack((inds_top_cent, inds_top_sats))

# get counts for the FoF TNG version
inds_gals = (np.argsort(SubhaloMstar_fp)[::-1])[:num_gals]
group_gals = SubhaloGrNr[inds_gals]
sub_grnr_uni, inds = np.unique(SubhaloGrNr, return_index=True)

# stellar mass of the central galaxy
GroupMstar = np.zeros(len(Group_M_Mean200))
GroupMstar[sub_grnr_uni] = SubhaloMstar_fp[inds]

# intersection between the selected galaxies and all the first subhalos
inds_cent, comm1, comm2 = np.intersect1d(inds_gals, inds, return_indices=True)
print("number of centrals = ", len(inds_cent))

# stellar mass of the central galaxy and halo mass
ms = SubhaloMstar_fp[inds_cent]
mh = Group_M_Mean200[sub_grnr_uni[comm2]]

# stellar mass of the central galaxy and halo mass
#ms = SubhaloMstar_fp[inds_gals]
#mh = Group_M_Mean200[group_gals]

plt.figure(figsize=(9, 7))
#plt.scatter(Group_M_Mean200[Group_M_Mean200 > 1.e11], (GroupMstar/Group_M_Mean200)[Group_M_Mean200 > 1.e11], marker='o', color=color1_tng, s=1, alpha=0.1)
plt.scatter(mh, ms/mh, marker='*', color=color2_tng, s=20, alpha=0.3)

bins = np.logspace(11, 15, 31)
binc = (bins[1:] + bins[:-1])*0.5

hist_gals, bins = np.histogram(mh, bins=bins, weights=ms/mh)
hist, bins = np.histogram(Group_M_Mean200[Group_M_Mean200 > 1.e11], bins=bins, weights=(GroupMstar/Group_M_Mean200)[Group_M_Mean200 > 1.e11])
hist_norm, bins = np.histogram(Group_M_Mean200, bins=bins)
std_gr, _, _ = stats.binned_statistic(Group_M_Mean200[Group_M_Mean200 > 1.e11], (GroupMstar/Group_M_Mean200)[Group_M_Mean200 > 1.e11], statistic='std', bins=bins)
#std_gr_hi, _, _ = stats.binned_statistic(Group_M_Mean200[Group_M_Mean200 > 1.e11], (GroupMstar/Group_M_Mean200)[Group_M_Mean200 > 1.e11], statistic=lambda x: np.percentile(x, q = 84.), bins=bins)
#std_gr_lo, _, _ = stats.binned_statistic(Group_M_Mean200[Group_M_Mean200 > 1.e11], (GroupMstar/Group_M_Mean200)[Group_M_Mean200 > 1.e11], statistic=lambda x: np.percentile(x, q = 16.), bins=bins)
hist_gr = hist/hist_norm
hist_gals_gr = hist_gals/hist_norm

ms = mstar[inds_top_cent]
mh = mvir[inds_top_cent]

hist_gals, bins = np.histogram(mh, bins=bins, weights=ms/mh)
hist, bins = np.histogram(halo_mvir, bins=bins, weights=(halo_mstar/halo_mvir))
hist_norm, bins = np.histogram(halo_mvir, bins=bins)
std, _, _ = stats.binned_statistic(halo_mvir, (halo_mstar/halo_mvir), statistic='std', bins=bins)
#std_hi, _, _ = stats.binned_statistic(halo_mvir, (halo_mstar/halo_mvir), statistic=lambda x: np.percentile(x, q = 84.), bins=bins)
#std_lo, _, _ = stats.binned_statistic(halo_mvir, (halo_mstar/halo_mvir), statistic=lambda x: np.percentile(x, q = 16.), bins=bins)
hist /= hist_norm
hist_gals /= hist_norm



#plt.scatter(halo_mvir[halo_mvir > 1.e11], (halo_mstar/halo_mvir)[halo_mvir > 1.e11], marker='*', color=color1_sam, s=1, alpha=0.1)
plt.scatter(mh, ms/mh, marker='*', color=color2_sam, s=20, alpha=0.3)
plt.plot(binc, hist_gr, color=color3_tng, label='TNG', lw=2.5)
plt.fill_between(binc, hist_gr+std_gr, hist_gr-std_gr, color=color3_tng, alpha=0.3)
#plt.fill_between(binc, std_gr_hi, std_gr_lo, color=color3_tng, alpha=0.3)
plt.plot(binc, hist_gals_gr, color=color3_tng, ls='--', lw=2.5)
plt.plot(binc, hist, color=color3_sam, label='SAM', lw=2.5)
plt.fill_between(binc, hist+std, hist-std, color=color3_sam, alpha=0.3)
#plt.fill_between(binc, std_hi, std_lo, color=color3_sam, alpha=0.3)
plt.plot(binc, hist_gals, color=color3_sam, ls='--', lw=2.5)


# universe machine
new_mach = True
if new_mach:
    newmach = np.loadtxt("UniMach/smhm_a1.002312.dat")

    m_halo_18 = 10.**newmach[:, 0]*h
    shmr_med_18 = 10.**newmach[:, 1]

    plt.plot(m_halo_18[m_halo_18 < 10**14.8], shmr_med_18[m_halo_18 < 10**14.8], color='gray', lw=3.5, label='UniverseMachine DR1')
else:
    logmhalo, logshmr, dlogup, dlogdown = np.loadtxt("UniMach/c_smmr_then_z"+zuni+"_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat", unpack=True)
    m_halo = 10.**logmhalo*h
    shmr_med = 10.**logshmr
    shmr_up = 10.**(logshmr+dlogup)
    shmr_down = 10.**(logshmr-dlogdown)
    plt.plot(m_halo, shmr_med, color='k', lw=1.5)
    plt.fill_between(m_halo, shmr_up, shmr_down, color='k', alpha=0.3)

plt.xlim([1.e11, 1.e15])
plt.ylim([1.e-4, 1.e-0])
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlabel(r'$M_{\rm halo}$')
plt.ylabel(r'$\langle M_{\rm \ast}/M_{\rm halo} \rangle$')
plt.savefig("figs/shmr_scatter"+str_snap+".png")
plt.show()
