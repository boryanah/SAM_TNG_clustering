import numpy as np
import matplotlib.pyplot as plt

#from tools.matching import get_match

sim_name = 'TNG300'
#sim_name = 'TNG100'
sam_dir = '/mnt/alan1/boryanah/SAM_subvolumes_'+sim_name+'/'

#snap_str = ''
snap_str = '_55'
#mass_type = 'Crit'
mass_type = 'Mean'

def get_dist(pos1,pos2):
    dx = np.abs(pos1-pos2)
    dx = np.where(dx>30.,dx-75.,dx)
    dist = np.sqrt(dx**2)
    return dist


def get_match(halo_subfind_id, SubGrNr_fp, N_halos_sam):
    # boolean array True for SAM halo matched with a subhalo in FP
    bool_matched = halo_subfind_id != -1

    print("percentage matched = ",np.sum(bool_matched)/len(bool_matched))

    # apply to isolate only the subhalo indices in Hydro that have been matched
    halo_subfind_id = halo_subfind_id[bool_matched]

    # get halo indices of the FP subhalos that have been matched
    hydro_matched = SubGrNr_fp[halo_subfind_id]
    #hydro_matched = halo_subfind_id
    
    # get the halo indices in the SAM
    inds_halo_sam = np.arange(N_halos_sam)
    sam_matched = inds_halo_sam[bool_matched]

    return hydro_matched, sam_matched

# load the galaxy arrays
hosthaloid = np.load(sam_dir+'GalpropHaloIndex'+snap_str+'.npy')
pos = np.load(sam_dir+'GalpropPos'+snap_str+'.npy')
N_subhalos_sam = hosthaloid.shape[0]
print("N subhalos", N_subhalos_sam)

#haloid = np.load(sam_dir+'HalopropIndex_Snapshot'+snap_str+'.npy')
halo_m_vir = np.load(sam_dir+'HalopropMvir'+snap_str+'.npy')
halo_c_nfw = np.load(sam_dir+'HalopropC_nfw'+snap_str+'.npy')
N_halos_sam = len(halo_m_vir)
print("N halos", N_halos_sam)


unique_hosts, inds_halo = np.unique(hosthaloid, return_index=True)
sat_type = np.ones(N_subhalos_sam, dtype=int)
sat_type[inds_halo] = 0
# TESTING this array not working
#sat_type = np.load(sam_dir+'GalpropSatType'+snap_str+'.npy')
halo_pos = pos[sat_type==0]

assert np.sum(sat_type==0) == halo_pos.shape[0], "whyyyy"
assert np.sum(sat_type==0) == N_halos_sam, "whaaat"


hydro_dir = '/mnt/gosling1/boryanah/'+sim_name+'/'
SubGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp'+snap_str+'.npy')
SubGrNr_dm = np.load(hydro_dir+'SubhaloGrNr_dm'+snap_str+'.npy')
SubMT_fp = np.load(hydro_dir+'SubhaloMassType_fp'+snap_str+'.npy')
GrMcrit_fp = np.load(hydro_dir+'Group_M_'+mass_type+'200_fp'+snap_str+'.npy')*1.e10
GrMcrit_dm = np.load(hydro_dir+'Group_M_'+mass_type+'200_dm'+snap_str+'.npy')*1.e10
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp'+snap_str+'.npy')/1.e3
GroupPos_dm = np.load(hydro_dir+'GroupPos_dm'+snap_str+'.npy')/1.e3

# indices swapped in the TNG100 data
FP_label = 'FP'
DM_label = 'DM'

# those two should be equivalent
halo_subfind_id = np.load(sam_dir+'GalpropSubfindIndex_'+FP_label+snap_str+'.npy')[sat_type==0]
# WHAT IS HAPPENING HERE
halo_subfind_id2 = np.load(sam_dir+'HalopropSubfindID'+snap_str+'.npy')
#print("this should be 0 = ", (halo_subfind_id-halo_subfind_id2).sum())

# these should also be equivalent
hydro_matched, sam_matched = get_match(halo_subfind_id, SubGrNr_fp, N_halos_sam)
halo_fof_index = np.load(sam_dir+'HalopropFoFIndex_'+FP_label+snap_str+'.npy')
halo_fof_index = halo_fof_index[halo_fof_index != -1]
print("this should be 0", np.sum(hydro_matched-halo_fof_index))

# these should be small
dist = get_dist(GroupPos_fp[hydro_matched],halo_pos[sam_matched])
print("max min dist between matched halos = ", dist.min(),dist.max())
print("printing some halo positions = ", GroupPos_fp[hydro_matched[200:210]],halo_pos[sam_matched[200:210]])


halo_subfind_id_dm = np.load(sam_dir+'GalpropSubfindIndex_'+DM_label+snap_str+'.npy')[sat_type==0]
hydro_matched_dm, sam_matched_dm = get_match(halo_subfind_id_dm, SubGrNr_dm, N_halos_sam)
GroupConc_nfw_dm = np.zeros(GroupPos_dm.shape[0])
GroupConc_nfw_dm[hydro_matched_dm] = halo_c_nfw[sam_matched_dm]
np.save('GroupConc_nfw_dm'+snap_str+'.npy', GroupConc_nfw_dm)
GroupConc_nfw_fp = np.zeros(GroupPos_fp.shape[0])
GroupConc_nfw_fp[hydro_matched] = halo_c_nfw[sam_matched]
np.save('GroupConc_nfw_fp'+snap_str+'.npy', GroupConc_nfw_fp)
print('fraction of halos that have matches = ', np.sum((GroupConc_nfw_fp > 0) & (GrMcrit_fp > 1.e11))/len(GroupConc_nfw_fp[GrMcrit_fp > 1.e11]))
print('fraction of halos that have matches = ', np.sum((GroupConc_nfw_dm > 0) & (GrMcrit_dm > 1.e11))/len(GroupConc_nfw_dm[GrMcrit_dm > 1.e11]))
GroupConc_fp = np.load(hydro_dir+'GroupConc_fp'+snap_str+'.npy')
GroupConc_dm = np.load(hydro_dir+'GroupConc_dm'+snap_str+'.npy')
print('fraction of halos that have NaN concentration = ', np.sum(np.isnan(GroupConc_fp) & (GrMcrit_fp > 1.e11))/len(GroupConc_fp))
print('fraction of halos that have NaN concentration = ', np.sum(np.isnan(GroupConc_dm) & (GrMcrit_dm > 1.e11))/len(GroupConc_dm))

print("min/max conc = ", GroupConc_nfw_dm[GroupConc_nfw_dm > 0.].min(), GroupConc_nfw_dm[GroupConc_nfw_dm > 0.].max())
print("min/max conc = ", GroupConc_nfw_fp[GroupConc_nfw_fp > 0.].min(), GroupConc_nfw_fp[GroupConc_nfw_fp > 0.].max())
print("min/max conc = ", GroupConc_dm[~ np.isnan(GroupConc_dm)].min(), GroupConc_dm[~ np.isnan(GroupConc_dm)].max())
print("min/max conc = ", GroupConc_fp[~ np.isnan(GroupConc_fp)].min(), GroupConc_fp[~ np.isnan(GroupConc_fp)].max())

plt.figure(1)
#plt.scatter(GrMcrit_fp[GroupConc_nfw_fp > 0], GroupConc_nfw_fp[GroupConc_nfw_fp > 0], s=0.1, alpha=0.1)
plt.scatter(GrMcrit_fp[~ np.isnan(GroupConc_fp)], 10.**GroupConc_fp[~ np.isnan(GroupConc_fp)], s=0.1, alpha=0.1)
plt.xscale('log')
plt.xlim([1.e11, 1.e15])
plt.ylim([0., 80])

plt.figure(2)
#plt.scatter(GrMcrit_dm[~ np.isnan(GroupConc_dm)], GroupConc_nfw_dm[GroupConc_nfw_dm > 0], s=0.1, alpha=0.1)
plt.scatter(GrMcrit_dm[~ np.isnan(GroupConc_dm)], 10.**GroupConc_dm[~ np.isnan(GroupConc_dm)], s=0.1, alpha=0.1)
plt.xscale('log')
plt.xlim([1.e11, 1.e15])
plt.ylim([0., 80])
plt.show()
quit()


hosts, inds, counts = np.unique(hosthaloid,return_index=True, return_counts=True)

assert np.sum(sat_type==0) == len(hosts), 'number of centrals'
assert np.sum(np.sort(hosthaloid)-hosthaloid) == 0, 'halo indices are ordered'

halo_pos = pos[sat_type==0]
halo_x = np.repeat(halo_pos[:,0],counts)
halo_y = np.repeat(halo_pos[:,1],counts)
halo_z = np.repeat(halo_pos[:,2],counts)
halo_pos = np.vstack((halo_x,halo_y,halo_z)).T

dist = get_dist(halo_pos,pos)
print(dist.max())
 
