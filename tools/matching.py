import numpy as np
from tools.match_searchsorted import match

def get_match(halo_subfind_id,SubGrNr_fp,N_halos_sam):
    # boolean array True for SAM halo matched with a subhalo in FP
    bool_matched = halo_subfind_id != -1

    # apply to isolate only the subhalo indices in FP that have been matched
    halo_subfind_id = halo_subfind_id[bool_matched]

    # get halo indices of the FP subhalos that have been matched
    hydro_matched = SubGrNr_fp[halo_subfind_id]

    # get the halo indices in the SAM
    inds_halo_sam = np.arange(N_halos_sam)
    sam_matched = inds_halo_sam[bool_matched]

    return hydro_matched, sam_matched

def match_halos(halo_fp,halo_sam,hydro_matched,sam_matched):
    halo_fp_matched = halo_fp[hydro_matched]
    halo_sam_matched = halo_sam[sam_matched]
    return halo_fp_matched, halo_sam_matched

def match_subs_sonya(inds_top, hosthaloid, sam_matched, halo_matched):
    host_inds_top = hosthaloid[inds_top]

    # assuming that all the galaxies in the halo which have been matched can be safety kept.
    matchedTNGhalos = halo_matched
    boolean = np.in1d(host_inds_top, matchedTNGhalos)
    matchedTNGgals_index = inds_top[boolean]
    halo_hosts = host_inds_top[boolean]
    print(halo_hosts[:100])
    #return matchedTNGgals_index, halo_hosts

    arr = match(host_inds_top, halo_matched)
    halo_hosts = host_inds_top[arr >= 0]
    sub_matched = inds_top[arr >= 0]
    sam_host_matched = arr[arr >= 0]
    #inds_host_matched = sam_matched[arr[arr >= 0]]
    print(halo_hosts[:100])
    
    return sub_matched, sam_host_matched, halo_hosts

def match_subs(inds_top, hosthaloid, sam_matched, halo_matched):
    host_inds_top = hosthaloid[inds_top]

    arr = match(host_inds_top, halo_matched)
    halo_hosts = host_inds_top[arr >= 0]
    sub_matched = inds_top[arr >= 0]
    sam_host_matched = arr[arr >= 0]
    #inds_host_matched = sam_matched[arr[arr >= 0]]

    return sub_matched, sam_host_matched, halo_hosts


def main():
    Lbox = 75 # Mpc/h

    # load SAM
    SAM_dir = '/mnt/store1/boryanah/SAM_subvolumes_TNG100/'
    hosthaloid = np.load(SAM_dir+'GalpropHaloIndex_corr.npy').astype(int)
    mhalo = np.load(SAM_dir+'GalpropMhalo.npy')
    mstar = np.load(SAM_dir+'GalpropMstar.npy')
    sat_type = np.load(SAM_dir+'GalpropSatType.npy')
    xyz_position = np.load(SAM_dir+'GalpropPos.npy')

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

    # load the matches with hydro
    halo_subfind_id = np.load(SAM_dir+'HalopropSubfindID.npy')#

    # obtain matching indices for halos
    hydro_matched, sam_matched = get_match(halo_subfind_id,SubGrNr_fp,N_halos_sam)

    # match halos in FP and SAM
    #GroupPos_fp, halo_xyz_position = match_halos(GroupPos_fp,halo_xyz_position,hydro_matched,sam_matched)
    print(GroupPos_fp[:10],halo_xyz_position[:10])

    # intersect top hosts with hydro and sam matched
    # print out no. gals

