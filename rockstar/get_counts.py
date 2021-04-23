import numpy as np
import matplotlib.pyplot as plt


# sim params
sam_dir = '/mnt/alan1/boryanah/SAM_subvolumes_TNG300/'
hydro_dir = '/mnt/gosling1/boryanah/TNG300/'
str_snap = ''
Lbox = 205.
fac = 2
#gal_type = 'mstar'
gal_type = 'sfr'
if gal_type == 'mstar':
    f_sat = 0.25
    num_gals = 12000
if gal_type == 'sfr':
    f_sat = 0.
    num_gals = 8000

# loading SAM
sat_type = np.load(sam_dir+'GalpropSatType'+str_snap+'.npy')
halo_mvir = np.load(sam_dir+'GalpropMhalo'+str_snap+'.npy')[sat_type == 0]
mstar = np.load(sam_dir+'GalpropMstar'+str_snap+'.npy')
sfr = np.load(sam_dir+'GalpropSfr'+str_snap+'.npy')
#sfr = np.load(sam_dir+'GalpropSfrave100myr'+str_snap+'.npy')
#hosthaloid = np.load(sam_dir+'GalpropHaloIndex_new'+str_snap+'.npy').astype(int) # use .get in marvin for the dictionaries also write it clearly and don't call new for snap 55
hosthaloid = np.load(sam_dir+'GalpropHaloIndex'+str_snap+'.npy').astype(int) # use .get in marvin for the dictionaries also write it clearly and don't call new

# abundance matching selection for the SAM move lower but note the sorting of mvir
if gal_type == 'mstar':
    inds_top = (np.argsort(mstar)[::-1])[:num_gals*fac]
elif gal_type == 'sfr':
    inds_top = (np.argsort(sfr)[::-1])[:num_gals*fac]
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

# sort all arrays in order of halo mass
i_sort = np.argsort(halo_mvir)[::-1]
halo_mvir = halo_mvir[i_sort]

# loading TNG 
SubhaloMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp'+str_snap+'.npy')[:, 4]*1.e10
SubhaloSFR_fp = np.load(hydro_dir+'SubhaloSFR_fp'+str_snap+'.npy')
SubhaloHaloNr = np.load("SubhaloHaloNr_fp"+str_snap+".npy")
SubhaloGrNr = np.load(hydro_dir+"SubhaloGrNr_fp"+str_snap+".npy")
#Group_M_Mean200 = np.load(hydro_dir+"Group_M_Mean200_fp"+str_snap+".npy")*1.e10 # weird ass problem # FIIIIIIIIIIIX MEEEEE
Group_M_Mean200 = np.load(hydro_dir+"Group_M_TopHat200_fp"+str_snap+".npy")*1.e10

# find the central subhalos
unique_hosts, inds_cent = np.unique(SubhaloGrNr,return_index=True)

# select galaxies
if gal_type == 'mstar':
    inds_gals = (np.argsort(SubhaloMstar_fp)[::-1])[:num_gals*fac]
elif gal_type == 'sfr':
    inds_gals = (np.argsort(SubhaloSFR_fp)[::-1])[:num_gals*fac]
inds_sats = inds_gal[np.in1d(inds_gal, inds_cent, invert=True)]
bool_gals_cent = np.in1d(inds_gals, inds_cent)
bool_gals_sats = np.in1d(inds_gals, inds_sats)
num_sats = int(np.round(f_sat*num_gals))
num_cent = num_gals - num_sats
inds_gals_cent = (inds_gals[bool_gals_cent])[:num_cent]
inds_gals_sats = (inds_gals[bool_gals_sats])[:num_sats]
inds_gals = np.hstack((inds_gals_cent, inds_gals_sats))

# get counts for the ROCKSTAR TNG version
host_gals = hosthaloid[inds_top]
host_gals_uni, host_cts_gals = np.unique(host_gals, return_counts=True)
host_counts = np.zeros(len(halo_mvir), dtype=int)
host_counts[host_gals_uni] = host_cts_gals
print("number of galaxies ROCKSTAR-SAM = ", np.sum(host_counts), np.max(host_counts))
np.save("halo_counts_"+gal_type+"_sam.npy", host_counts)

# get counts for the ROCKSTAR TNG version
halo_gals = SubhaloHaloNr[inds_gals]
halo_gals = halo_gals[halo_gals > -1]
print(len(halo_gals))
halo_gals_uni, halo_cts_gals = np.unique(halo_gals, return_counts=True)
halo_counts = np.zeros(len(halo_mvir), dtype=int)
halo_counts[halo_gals_uni] = halo_cts_gals
print("number of galaxies ROCKSTAR-TNG = ", np.sum(halo_counts), np.max(halo_counts))
np.save("halo_counts_"+gal_type+"_tng.npy", halo_counts)

# get counts for the FoF TNG version
group_gals = SubhaloGrNr[inds_gals]
group_gals_uni, group_cts_gals = np.unique(group_gals, return_counts=True)
group_counts = np.zeros(len(Group_M_Mean200), dtype=int)
group_counts[group_gals_uni] = group_cts_gals
print("number of galaxies FoF-TNG = ", np.sum(group_counts), np.max(group_counts))
np.save("group_counts_"+gal_type+"_tng.npy", group_counts)
