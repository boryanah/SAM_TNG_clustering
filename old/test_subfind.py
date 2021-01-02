import numpy as np


def main():
    Lbox = 75 # Mpc/h

    # load SAM
    SAM_dir = '/mnt/store1/boryanah/SAM_subvolumes/'
    hosthaloid = np.load(SAM_dir+'GalpropHaloIndex_corr.npy').astype(int)
    mhalo = np.load(SAM_dir+'GalpropMhalo.npy')
    mstar = np.load(SAM_dir+'GalpropMstar.npy')
    sat_type = np.load(SAM_dir+'GalpropSatType.npy')
    xyz_position = np.load(SAM_dir+'GalpropPos.npy')

    # load the matches with hydro
    halo_subfind_id = np.load(SAM_dir+'HalopropSubfindID.npy')

    # find the halo positions
    xyz_position[xyz_position > Lbox] -= Lbox
    xyz_position[xyz_position < 0.] += Lbox 
    halo_xyz_position = xyz_position[sat_type == 0]

    # number of halos in SAM (equivalent to length of any of the Haloprop arrays and the number of sat_type == 0 objects)
    N_halos_sam = halo_xyz_position.shape[0]

    # load hydro
    hydro_dir = '/mnt/gosling1/boryanah/TNG100/'
    SubGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp.npy')
    GrMcrit_fp = np.load(hydro_dir+'Group_M_Crit200_fp.npy')*1.e10
    GroupPos_fp = np.load(hydro_dir+'GroupPos_fp.npy')/1000.
    SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp.npy')/1000.

main()

# this is a boolean array with True for every SAM halo that has been matched with a subhalo in FP
bool_matched = halo_subfind_id != -1

# we apply that array to isolate only the subhalo indices in FP that have been matched
halo_subfind_id = halo_subfind_id[bool_matched]

# we can now get the halo indices of the FP subhalos that have been matched
hydro_matched = SubGrNr_fp[halo_subfind_id]

# we can also get the halo indices in the SAM
inds_halo_sam = np.arange(N_halos_sam)
sam_matched = inds_halo_sam[bool_matched]

# let's now check how that works:

# take the matched halos in FP and SAM (first index in SAM corresponds to first in hydro, second to second, etc.)
GroupPos_fp_matched = GroupPos_fp[hydro_matched]
halo_xyz_position_matched = halo_xyz_position[sam_matched]

# select 100 (just random) halo positions in FP and SAM and check whether they match well (they shouldn't be perfect because the halo finders are still different)
for i in range(100):
    print("halo position: SAM and Hydro",halo_xyz_position_matched[i],GroupPos_fp_matched[i])
    
# we need to now apply these sam_matched and hydro_matched to all halo arrays in hydro and all halo arrays in SAM

# can also maybe check how well the halo masses are matched

# unique halo hosts
unique_hosts = np.unique(SubGrNr_fp)

print(len(bool_matched),len(unique_hosts),len(GrMcrit_fp))

# some useful statistics (feel free to ignore)
print("percentage matched in SAM: ",np.sum(bool_matched)/len(bool_matched)*100.)
print("percentage matched in Hydro: ",np.sum(bool_matched)/len(unique_hosts)*100.)


quit()
i_start = 100000
i_end = i_start+10
print(GroupPos_fp[sub_matched][i_start:i_end])
print("vs.")
print(halo_xyz_position[sam_matched][i_start:i_end])
'''
TNG_centrals = TNG_subfind.iloc[SAM_centrals['subfind-idx']].reset_index(drop=False) 
TNG_centrals['sam-idx'] = SAM_centrals.index.values
TNG_centrals = TNG_centrals[TNG_centrals['central']].reset_index(drop=True) # double check if TNG is a central
SAM_centrals.iloc[TNG_centrals['sam-idx']] # if some weren't centrals in TNG, we filter now filter them out in the SAM

'''
quit()

SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp.npy')[:,4]*1.e10

# total number of halos
N_halos_hydro = len(GrMcrit_fp)

# environment
halo_ijk = (GroupPos_fp/gr_size).astype(int)%n_gr
GroupEnv_fp = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

GrRcrit_fp = np.load(hydro_dir+'Group_R_Crit200_fp.npy')
Group_Vmax_fp = np.load(hydro_dir+'Group_Vmax_fp.npy')/1000.
#Group_R_Mean200_fp = np.load(hydro_dir+'Group_R_Mean200_fp.npy')/1000
Group_V_Crit200_fp = np.load(hydro_dir+'Group_V_Crit200_fp.npy')/1000
SubhaloSpin_fp = np.sqrt(np.sum(np.load(hydro_dir+'SubhaloSpin_fp.npy')**2,axis=1))
SubhaloVelDisp_fp = np.load(hydro_dir+'SubhaloVelDisp_fp.npy')
SubhaloHalfmassRad_fp = np.load(hydro_dir+'SubhaloHalfmassRad_fp.npy')
