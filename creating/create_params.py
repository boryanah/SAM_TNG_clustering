import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.ndimage import gaussian_filter
from tools.halostats import get_hist_count

#N_dim = 128
N_dim = 256
#Lbox = 75.
Lbox = 205.
#num_gals_hydro = 6000
num_gals_hydro = 12000
#hydro_dir = '/mnt/gosling1/boryanah/TNG100/'
hydro_dir = '/mnt/gosling1/boryanah/TNG300/'
#mass_type = 'Crit'
mass_type = 'Mean'
snap_str = '_55'
#snap_str = ''
#gal_type = 'Mass'
gal_type = 'ELG_DESI'

# fp dmo matching
# TESTING original old
#fp_dmo_halo_inds = np.load(hydro_dir+"fp_dmo_inds_halo_2500.npy")#.T
#fp_halo_inds = fp_dmo_halo_inds[:,0]
#dmo_halo_inds = fp_dmo_halo_inds[:,1]
#mag_exc = 2063
#fp_halo_inds = fp_halo_inds[:-mag_exc]
#dmo_halo_inds = dmo_halo_inds[:-mag_exc]
# new algorithm
fp_dmo_halo_inds = np.load(hydro_dir+"fp_dmo_halo_inds.npy")
fp_halo_inds = fp_dmo_halo_inds[0]
dmo_halo_inds = fp_dmo_halo_inds[1]

# directory in which to save output
save_hydro_dir = ''


# you can try to write your own version of this
# if you give this function an (N_particles,3) numpy array, it gives back the density for a given Lbox and cell number N_dim
def get_density(pos,N_dim=128,Lbox=75.):
    # x, y, and z position
    g_x = pos[:,0]
    g_y = pos[:,1]
    g_z = pos[:,2]
    # total number of objects
    N_g = len(g_x)
    # get a 3d histogram with number of objects in each cell
    D, edges = np.histogramdd(np.transpose([g_x,g_y,g_z]),bins=N_dim,range=[[0,Lbox],[0,Lbox],[0,Lbox]])
    # average number of particles per cell
    D_avg = N_g*1./N_dim**3
    D /= D_avg
    D -= 1.
    return D

if not os.path.exists(hydro_dir+"smoothed_mass_in_area"+snap_str+".npy"):
    if snap_str == '':
        pos_parts = np.load("/mnt/gosling1/boryanah/TNG300/parts_position_tng300-3_99.npy")/1.e3#"particles are not here"
    elif snap_str == '_55':
        pos_parts = np.load("/mnt/gosling1/boryanah/TNG300/pos_parts_tng300-3_55.npy")/1.e3
    print(np.max(pos_parts[:,0]))
    
    # call function above for getting density
    D_g = get_density(pos_parts,N_dim,Lbox)

    # smoothing
    R = 1.1#2. # Mpc/h
    D_g_smo = gaussian_filter(D_g,R)

    np.save(hydro_dir+"smoothed_mass_in_area"+snap_str+".npy",D_g_smo)

# environment parameter
density = np.load(hydro_dir+'smoothed_mass_in_area'+snap_str+'.npy')
n_gr = density.shape[0]
print("n_gr = ",n_gr)
gr_size = Lbox/n_gr

# Loading data for the hydro -- you can skip for now
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp'+snap_str+'.npy')/1000.
Group_M_Crit200_fp = np.load(hydro_dir+'Group_M_'+mass_type+'200_fp'+snap_str+'.npy')
SubhaloGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp'+snap_str+'.npy')
SubhaloVmax_fp = np.load(hydro_dir+'SubhaloVmax_fp'+snap_str+'.npy')
if not os.path.exists(hydro_dir+'Group_Vmax_fp'+snap_str+'.npy'):
    unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)
    Group_Vmax_fp = np.zeros_like(Group_M_Crit200_fp)
    Group_Vmax_fp[unique_sub_grnr] = SubhaloVmax_fp[firsts]
    np.save(hydro_dir+"Group_Vmax_fp"+snap_str+".npy", Group_Vmax_fp)
Group_Vmax_fp = np.load(hydro_dir+'Group_Vmax_fp'+snap_str+'.npy')
Group_R_Crit200_fp = np.load(hydro_dir+'Group_R_'+mass_type+'200_fp'+snap_str+'.npy')
if not os.path.exists(hydro_dir+'Group_V_'+mass_type+'200_fp'+snap_str+'.npy'):
    G_N = 4.302e-3
    print("saving")
    Group_V_Crit200_fp = np.sqrt(G_N*Group_M_Crit200_fp*1.e10/(Group_R_Crit200_fp*1.e3))
    print(Group_V_Crit200_fp[:100])
    np.save(hydro_dir+'Group_V_'+mass_type+'200_fp'+snap_str+'.npy',Group_V_Crit200_fp)
Group_V_Crit200_fp = np.load(hydro_dir+'Group_V_'+mass_type+'200_fp'+snap_str+'.npy')
Group_M_Mean200_fp = np.load(hydro_dir+'Group_M_Mean200_fp'+snap_str+'.npy')
SubhaloSpin_fp = np.load(hydro_dir+'SubhaloSpin_fp'+snap_str+'.npy')
SubhaloSpin_fp = np.sum(SubhaloSpin_fp,axis=1)
SubhaloVelDisp_fp = np.load(hydro_dir+'SubhaloVelDisp_fp'+snap_str+'.npy')
#SubhaloHalfmassRad_fp = np.load(hydro_dir+'SubhaloHalfmassRad_fp'+snap_str+'.npy')

# only full-physics
SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp'+snap_str+'.npy')[:,4]*1.e10
SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp'+snap_str+'.npy')/1000.

GroupPos_dm = np.load(hydro_dir+'GroupPos_dm'+snap_str+'.npy')/1000.
Group_M_Crit200_dm = np.load(hydro_dir+'Group_M_'+mass_type+'200_dm'+snap_str+'.npy')
SubhaloGrNr_dm = np.load(hydro_dir+'SubhaloGrNr_dm'+snap_str+'.npy')
SubhaloVmax_dm = np.load(hydro_dir+'SubhaloVmax_dm'+snap_str+'.npy')
if not os.path.exists(hydro_dir+'Group_Vmax_dm'+snap_str+'.npy'):
    unique_sub_grnr, firsts = np.unique(SubhaloGrNr_dm,return_index=True)
    Group_Vmax_dm = np.zeros_like(Group_M_Crit200_dm)
    Group_Vmax_dm[unique_sub_grnr] = SubhaloVmax_dm[firsts]
    np.save(hydro_dir+"Group_Vmax_dm"+snap_str+".npy", Group_Vmax_dm)
Group_Vmax_dm = np.load(hydro_dir+'Group_Vmax_dm'+snap_str+'.npy')
SubhaloVelDisp_dm = np.load(hydro_dir+'SubhaloVelDisp_dm'+snap_str+'.npy')
Group_R_Crit200_dm = np.load(hydro_dir+'Group_R_'+mass_type+'200_dm'+snap_str+'.npy')
Group_M_Mean200_dm = np.load(hydro_dir+'Group_M_Mean200_dm'+snap_str+'.npy')
if not os.path.exists(hydro_dir+'Group_V_'+mass_type+'200_dm'+snap_str+'.npy'):
    G_N = 4.302e-3
    Group_V_Crit200_dm = np.sqrt(G_N*Group_M_Crit200_dm*1.e10/(Group_R_Crit200_dm*1.e3))
    np.save(hydro_dir+'Group_V_'+mass_type+'200_dm'+snap_str+'.npy',Group_V_Crit200_dm)
Group_V_Crit200_dm = np.load(hydro_dir+'Group_V_'+mass_type+'200_dm'+snap_str+'.npy')
SubhaloSpin_dm = np.load(hydro_dir+'SubhaloSpin_dm'+snap_str+'.npy')
SubhaloSpin_dm = np.sum(SubhaloSpin_dm,axis=1)
SubhaloHalfmassRad_dm = np.load(hydro_dir+'SubhaloHalfmassRad_dm'+snap_str+'.npy')

# select galaxies
if gal_type == 'Mass':
    inds_top = np.argsort(SubMstar_fp)[::-1][:num_gals_hydro]
elif gal_type == 'ELG_DESI':
    inds_top = np.load(save_hydro_dir+"SubhaloIndsPreELG_DESI_fp"+snap_str+".npy")
inds_host_top = SubhaloGrNr_fp[inds_top]

# obtain count_fp
hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top, inds_host_top, SubhaloPos_fp, Group_M_Crit200_fp, GroupPos_fp, Lbox=Lbox, record_relative=True)

# select halos with 1 or more galaxies
inds_halo_fp = np.arange(len(Group_M_Crit200_fp),dtype=int)
inds_count_fp = inds_halo_fp[count_fp > 0]

# how many of them have matches
inter, comm1, comm2 = np.intersect1d(inds_count_fp,fp_halo_inds,return_indices=True)
print("matches for = ",len(comm2)*100./len(inds_count_fp))

# translate counts to DMO
count_dm = np.zeros(len(Group_M_Crit200_dm),dtype=int)
count_dm[dmo_halo_inds[comm2]] = count_fp[inter]
np.save(save_hydro_dir+"GroupCount"+gal_type+"_fp"+snap_str+".npy",count_fp)
np.save(save_hydro_dir+"GroupCount"+gal_type+"_dm"+snap_str+".npy",count_dm)
#print(GroupPos_dm[dmo_halo_inds[comm2]],GroupPos_fp[inter])
print(GroupPos_dm[dmo_halo_inds[comm2]][:100],GroupPos_fp[inter][:100])
print("sum of gals = ",np.sum(count_dm),np.sum(count_fp[inter]))

top_in_match = np.in1d(inds_host_top,inter)
print(np.sum(top_in_match))

inds_gal = inds_top[top_in_match]
np.save(save_hydro_dir+"SubhaloInds"+gal_type+"_fp"+snap_str+".npy",inds_gal)

unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)
GroupSpin_fp = np.zeros_like(Group_V_Crit200_fp)
GroupSpin_fp[unique_sub_grnr] = SubhaloSpin_fp[firsts]
GroupSpin_fp /= (Group_V_Crit200_fp*Group_M_Crit200_fp*Group_R_Crit200_fp)
np.save(save_hydro_dir+"GroupSpin_fp"+snap_str+".npy", GroupSpin_fp)
Group_VsqR_fp = np.zeros_like(Group_V_Crit200_fp)
#Group_VsqR_fp[unique_sub_grnr] = (SubhaloVelDisp_fp**2*SubhaloHalfmassRad_fp)[firsts]
#np.save(save_hydro_dir+"Group_VsqR_fp'+snap_str+'.npy", Group_VsqR_fp)
GroupConc_fp = (Group_Vmax_fp/Group_V_Crit200_fp)
np.save(save_hydro_dir+"GroupConc_fp"+snap_str+".npy", GroupConc_fp)

unique_sub_grnr, firsts = np.unique(SubhaloGrNr_dm,return_index=True)
GroupSpin_dm = np.zeros_like(Group_V_Crit200_dm)
GroupSpin_dm[unique_sub_grnr] = SubhaloSpin_dm[firsts]
GroupSpin_dm /= (Group_V_Crit200_dm*Group_M_Crit200_dm*Group_R_Crit200_dm)
np.save(save_hydro_dir+"GroupSpin_dm"+snap_str+".npy", GroupSpin_dm)
Group_VsqR_dm = np.zeros_like(Group_V_Crit200_dm)
Group_VsqR_dm[unique_sub_grnr] = (SubhaloVelDisp_dm**2*SubhaloHalfmassRad_dm)[firsts]
np.save(save_hydro_dir+"Group_VsqR_dm"+snap_str+".npy", Group_VsqR_dm)
GroupConc_dm = (Group_Vmax_dm/Group_V_Crit200_dm)
np.save(save_hydro_dir+"GroupConc_dm"+snap_str+".npy", GroupConc_dm)

# environment
halo_ijk = (GroupPos_dm/gr_size).astype(int)%n_gr
GroupEnv_dm = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

np.save(save_hydro_dir+"GroupEnv_dm"+snap_str+".npy", GroupEnv_dm)
