import numpy as np
import matplotlib.pyplot as plt
from tools.halostats import get_hod

# matplotlib settings
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times'],'size':18})
rc('text', usetex=True)
color_sam = 'dodgerblue'
color_tng = '#CC6677'
#gal_type = 'mstar'
gal_type = 'sfr'

import plotparams
plotparams.buba()

# sim params
sam_dir = '/mnt/alan1/boryanah/SAM_subvolumes_TNG300/'
hydro_dir = '/mnt/gosling1/boryanah/TNG300/'
str_snap = ''
Lbox = 205.

# loading SAM
sat_type = np.load(sam_dir+'GalpropSatType'+str_snap+'.npy')
halo_mvir = np.load(sam_dir+'GalpropMhalo'+str_snap+'.npy')[sat_type == 0]

# sort all arrays in order of halo mass
i_sort = np.argsort(halo_mvir)[::-1]
i_sort_rev = np.argsort(i_sort)
halo_mvir = halo_mvir[i_sort]
halo_counts = np.load("halo_counts_"+gal_type+"_tng.npy")
halo_counts_sam = np.load("halo_counts_"+gal_type+"_sam.npy") # note that those are in unsorted order

# loading TNG 
#Group_M_Mean200 = np.load(hydro_dir+"Group_M_Mean200_fp"+str_snap+".npy")*1.e10
Group_M_Mean200 = np.load(hydro_dir+"Group_M_TopHat200_fp"+str_snap+".npy")*1.e10
GroupCounts = np.load("group_counts_"+gal_type+"_tng.npy")

# hod
halo_hist_hod, bin_cents = get_hod(halo_mvir, halo_counts)
halo_sam_hist_hod, bin_cents = get_hod(halo_mvir[i_sort_rev], halo_counts_sam)
group_hist_hod, _ = get_hod(Group_M_Mean200, GroupCounts)

'''
# cum
# number of bin edges
n_bins = 31
# bin edges (btw Delta = np.log10(bins[1])-np.log10(bins[0]) in log space)
bins = np.logspace(10.,15.,n_bins)
# bin centers
bin_cents = 0.5*(bins[1:]+bins[:-1])
group_hist_hod, _ = np.histogram(Group_M_Mean200, bins=bins, weights=GroupCounts)
halo_hist_hod, _ = np.histogram(halo_mvir, bins=bins, weights=halo_counts)
halo_sam_hist_hod, _ = np.histogram(halo_mvir[i_sort_rev], bins=bins, weights=halo_counts_sam) # cause in unsorted order
'''

plt.figure(figsize=(9, 7))
plt.plot(bin_cents, halo_sam_hist_hod, color=color_sam, lw=2.5, label='SAM (ROCKSTAR)')
plt.plot(bin_cents, group_hist_hod, color=color_tng, lw=2.5, label='TNG (FoF)')
plt.plot(bin_cents, halo_hist_hod, color=color_tng, ls='--', lw=2.5, label='TNG (ROCKSTAR)')
plt.xscale('log')
plt.yscale('log')
if gal_type == 'mstar':
    plt.ylim([7.e-3,100.])
elif gal_type == 'sfr':
    plt.ylim([7.e-4,10.])
plt.xlim([2.e11,1.e15])
plt.legend(frameon=False)
plt.ylabel(r'$\langle N_{\rm gal} \rangle$')
plt.xlabel(r'$M_{\rm halo}$')
plt.savefig("figs/rockstar_hod_"+gal_type+".png")
plt.show()
