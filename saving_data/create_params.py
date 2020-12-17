import numpy as np
import matplotlib.pyplot as plt
import os

from tools.halostats import get_hist_count

N_dim = 128
Lbox = 75.
num_gals_hydro = 6000

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

if not os.path.exists("../smoothed_mass_in_area.npy"):
    pos_parts = "particles are not here"
    
    # call function above for getting density
    D_g = get_density(pos_parts,N_dim,Lbox)

    # smoothing
    R = 1.1#2. # Mpc/h
    D_g_smo = gaussian_filter(D_g,R)

    np.save("../smoothed_mass_in_area.npy",D_g_smo)

# environment parameter
density = np.load('../smoothed_mass_in_area.npy')
n_gr = density.shape[0]
print("n_gr = ",n_gr)
gr_size = Lbox/n_gr

# Loading data for the hydro -- you can skip for now
hydro_dir = '/mnt/gosling1/boryanah/TNG100/'
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp.npy')/1000.
SubhaloGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp.npy')
Group_Vmax_fp = np.load(hydro_dir+'Group_Vmax_fp.npy')
Group_V_Crit200_fp = np.load(hydro_dir+'Group_V_Crit200_fp.npy')
Group_R_Crit200_fp = np.load(hydro_dir+'Group_R_Crit200_fp.npy')
Group_M_Crit200_fp = np.load(hydro_dir+'Group_M_Crit200_fp.npy')
Group_M_Mean200_fp = np.load(hydro_dir+'Group_M_Mean200_fp.npy')
SubhaloSpin_fp = np.load(hydro_dir+'SubhaloSpin_fp.npy')
SubhaloSpin_fp = np.sum(SubhaloSpin_fp,axis=1)
SubhaloVelDisp_fp = np.load(hydro_dir+'SubhaloVelDisp_fp.npy')
SubhaloHalfmassRad_fp = np.load(hydro_dir+'SubhaloHalfmassRad_fp.npy')

SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp.npy')[:,4]*1.e10
SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp.npy')/1000.

GroupPos_dm = np.load(hydro_dir+'GroupPos_dm.npy')/1000.
SubhaloGrNr_dm = np.load(hydro_dir+'SubhaloGrNr_dm.npy')
Group_Vmax_dm = np.load(hydro_dir+'Group_Vmax_dm.npy')
SubhaloVelDisp_dm = np.load(hydro_dir+'SubhaloVelDisp_dm.npy')
Group_V_Crit200_dm = np.load(hydro_dir+'Group_V_Crit200_dm.npy')
Group_R_Crit200_dm = np.load(hydro_dir+'Group_R_Crit200_dm.npy')
Group_M_Crit200_dm = np.load(hydro_dir+'Group_M_Crit200_dm.npy')
Group_M_Mean200_dm = np.load(hydro_dir+'Group_M_Mean200_dm.npy')
SubhaloSpin_dm = np.load(hydro_dir+'SubhaloSpin_dm.npy')
SubhaloSpin_dm = np.sum(SubhaloSpin_dm,axis=1)
SubhaloHalfmassRad_dm = np.load(hydro_dir+'SubhaloHalfmassRad_dm.npy')


# fp dmo matching
fp_dmo_halo_inds = np.load(hydro_dir+"fp_dmo_halo_inds.npy")
fp_halo_inds = fp_dmo_halo_inds[0]
dmo_halo_inds = fp_dmo_halo_inds[1]

# select galaxies
inds_top = np.argsort(SubMstar_fp)[::-1][:num_gals_hydro]
inds_host_top = SubhaloGrNr_fp[inds_top]

# obtain count_fp
hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top, inds_host_top, SubhaloPos_fp, Group_M_Crit200_fp, GroupPos_fp, record_relative=True)

# select halos with 1 or more galaxies
inds_halo_fp = np.arange(len(Group_M_Crit200_fp),dtype=int)
inds_count_fp = inds_halo_fp[count_fp > 0]

# how many of them have matches
inter, comm1, comm2 = np.intersect1d(inds_count_fp,fp_halo_inds,return_indices=True)
print("matches for = ",len(comm2)*100./len(inds_count_fp))

# translate counts to DMO
count_dm = np.zeros(len(Group_M_Crit200_dm),dtype=int)
count_dm[dmo_halo_inds[comm2]] = count_fp[inter]
np.save(hydro_dir+"GroupCountMass_fp.npy",count_fp[inter])
np.save(hydro_dir+"GroupCountMass_dm.npy",count_dm[inter])
print(GroupPos_dm[dmo_halo_inds[comm2]],GroupPos_fp[inter])
print("sum of gals = ",np.sum(count_dm),np.sum(count_fp[inter]))

top_in_match = np.in1d(inds_host_top,inter)
print(np.sum(top_in_match))

inds_gal = inds_top[top_in_match]
np.save(hydro_dir+"SubhaloIndsMass_fp.npy",inds_gal)

unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)
GroupSpin_fp = np.zeros_like(Group_V_Crit200_fp)
GroupSpin_fp[unique_sub_grnr] = SubhaloSpin_fp[firsts]
GroupSpin_fp /= (Group_V_Crit200_fp*Group_M_Crit200_fp*Group_R_Crit200_fp)
np.save(hydro_dir+"GroupSpin_fp.npy", GroupSpin_fp)
Group_VsqR_fp = np.zeros_like(Group_V_Crit200_fp)
Group_VsqR_fp[unique_sub_grnr] = (SubhaloVelDisp_fp**2*SubhaloHalfmassRad_fp)[firsts]
np.save(hydro_dir+"Group_VsqR_fp.npy", Group_VsqR_fp)
GroupConc_fp = (Group_Vmax_fp/Group_V_Crit200_fp)
np.save(hydro_dir+"GroupConc_fp.npy", GroupConc_fp)

unique_sub_grnr, firsts = np.unique(SubhaloGrNr_dm,return_index=True)
GroupSpin_dm = np.zeros_like(Group_V_Crit200_dm)
GroupSpin_dm[unique_sub_grnr] = SubhaloSpin_dm[firsts]
GroupSpin_dm /= (Group_V_Crit200_dm*Group_M_Crit200_dm*Group_R_Crit200_dm)
np.save(hydro_dir+"GroupSpin_dm.npy", GroupSpin_dm)
Group_VsqR_dm = np.zeros_like(Group_V_Crit200_dm)
Group_VsqR_dm[unique_sub_grnr] = (SubhaloVelDisp_dm**2*SubhaloHalfmassRad_dm)[firsts]
np.save(hydro_dir+"Group_VsqR_dm.npy", Group_VsqR_dm)
GroupConc_dm = (Group_Vmax_dm/Group_V_Crit200_dm)
np.save(hydro_dir+"GroupConc_dm.npy", GroupConc_dm)

# environment
halo_ijk = (GroupPos_fp/gr_size).astype(int)%n_gr
GroupEnv_fp = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

np.save(hydro_dir+"GroupEnv_dm.npy", GroupEnv_fp)
