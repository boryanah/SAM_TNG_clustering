import numpy as np
import matplotlib.pyplot as plt

sam_dir = '/mnt/store1/boryanah/SAM_subvolumes/'

snap_str = ''

def get_dist(pos1,pos2):
    dx = np.abs(pos1-pos2)
    dx = np.where(dx>30.,dx-75.,dx)
    dist = np.sqrt(dx**2)
    return dist

hosthaloid = np.load(sam_dir+"GalpropHaloIndex"+snap_str+".npy")
sat_type = np.load(sam_dir+"GalpropSatType"+snap_str+".npy")
#haloid = np.load(sam_dir+"HalopropIndex_Snapshot"+snap_str+".npy")
pos = np.load(sam_dir+"GalpropPos"+snap_str+".npy")
halo_m_vir = np.load(sam_dir+'HalopropMvir.npy')
halo_pos = pos[sat_type==0]
N_halos_sam = len(halo_m_vir)

# TESTING matching
from tools.matching import get_match

hydro_dir = '/mnt/gosling1/boryanah/TNG100/'
SubGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp.npy')
GrMcrit_fp = np.load(hydro_dir+'Group_M_Crit200_fp.npy')*1.e10
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp.npy')/1.e3

# those two should be equivalent
#halo_subfind_id = np.load(sam_dir+'GalpropSubfindIndex_FP.npy')[sat_type==0]
halo_subfind_id = np.load(sam_dir+'HalopropSubfindID.npy')
hydro_matched, sam_matched = get_match(halo_subfind_id,SubGrNr_fp,N_halos_sam)
halo_fof_index = np.load(sam_dir+'HalopropFoFIndex_FP.npy')
halo_fof_index = halo_fof_index[halo_fof_index != -1]
print(np.sum(hydro_matched-halo_fof_index))

dist = get_dist(GroupPos_fp[hydro_matched],halo_pos[sam_matched])
print(dist.min(),dist.max())
print(GroupPos_fp[hydro_matched[:10]],halo_pos[sam_matched[:10]])
quit()


hosts, inds, counts = np.unique(hosthaloid,return_index=True, return_counts=True)

assert np.sum(sat_type==0) == len(hosts), "number of centrals"
assert np.sum(np.sort(hosthaloid)-hosthaloid) == 0, "halo indices are ordered"

halo_pos = pos[sat_type==0]
halo_x = np.repeat(halo_pos[:,0],counts)
halo_y = np.repeat(halo_pos[:,1],counts)
halo_z = np.repeat(halo_pos[:,2],counts)
halo_pos = np.vstack((halo_x,halo_y,halo_z)).T

dist = get_dist(halo_pos,pos)
print(dist.max())
 
