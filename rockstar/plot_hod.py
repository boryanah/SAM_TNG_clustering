import numpy as np
import matplotlib.pyplot as plt
from tools.halostats import get_hod

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
halo_counts = np.load("halo_counts.npy")
halo_counts_sam = np.load("halo_counts_sam.npy") # note that those are in unsorted order

# loading TNG 
#Group_M_Mean200 = np.load(hydro_dir+"Group_M_Mean200_fp"+str_snap+".npy")*1.e10
Group_M_Mean200 = np.load(hydro_dir+"Group_M_TopHat200_fp"+str_snap+".npy")*1.e10
GroupCounts = np.load("group_counts.npy")

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

plt.plot(bin_cents, halo_hist_hod, label='TNG-ROCKSTAR')
plt.plot(bin_cents, halo_sam_hist_hod, label='SAM-ROCKSTAR')
plt.plot(bin_cents, group_hist_hod, label='TNG-FOF')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
