import matplotlib.pyplot as plt
import numpy as np
import plotparams # for making plots pretty, but comment out
plotparams.buba() # same
import Corrfunc # comment out since you're not using it
import sys

from tools.matching import get_match, match_halos, match_subs
from tools.halostats import get_from_sub_prop, get_shmr, get_shmr_prop, plot_shmr, plot_shmr_prop
    
# Hubble constant, snapshot number (99 is last snapshot, i.e. z = 0), number of galaxies
h = 0.6774
k = 99
num_gals_hydro = int(sys.argv[1])
rough_percent_matches = 98.6
num_gals_sam = int(rough_percent_matches/100*num_gals_hydro)
Lbox = 75.
n_gal = num_gals_sam/Lbox**3.
want_matching_sam = False
want_matching_hydro = True
record_relative = 1
type_gal = sys.argv[2]#"mstar"#"sfr"
try:
    secondary_property = sys.argv[3]
except:
    secondary_property = 'shuff'
    halo_prop = None
    group_prop = None
    order_type = 'mixed' # whatever
    sec_label = 'shuffled'

##########################################
#####               SAM              ##### 
##########################################

# Loading data for the SAM

#sam_data = 'recalibrated'
sam_data = 'recalibrated-updated'
if sam_data == 'recalibrated':
    # subhalo arrays
    SAM_dir = '/mnt/gosling1/boryanah/SAM/TNG100-1-SAM-Recalibrated/'
    hosthaloid = np.load(SAM_dir+'hosthaloid_%d.npy'%k).astype(int)
    mhalo = np.load(SAM_dir+'mhalo_%d.npy'%k)
    mstar = np.load(SAM_dir+'mstar_%d.npy'%k)
    sat_type = np.load(SAM_dir+'sat_type_%d.npy'%k)

    x_position = np.load(SAM_dir+'x_position_%d.npy'%k)
    y_position = np.load(SAM_dir+'y_position_%d.npy'%k)
    z_position = np.load(SAM_dir+'z_position_%d.npy'%k)

    # halo arrays
    halo_m_vir = np.load(SAM_dir+'halo_m_vir_%d.npy'%k)
    halo_c_nfw = np.load(SAM_dir+'halo_c_nfw_%d.npy'%k)
    halo_rhalo = np.load(SAM_dir+'halo_rhalo_%d.npy'%k)
    halo_spin = np.load(SAM_dir+'halo_spin_%d.npy'%k)
    halo_sigma_bulge = np.load(SAM_dir+'halo_sigma_bulge_%d.npy'%k)

    # combine the halo positions into one array
    xyz_position = np.vstack((x_position,y_position,z_position)).T
    xyz_position[xyz_position > Lbox] -= Lbox
    xyz_position[xyz_position < 0.] += Lbox 
    
elif sam_data == 'recalibrated-updated':
    # subhalo arrays
    SAM_dir = '/mnt/store1/boryanah/SAM_subvolumes/'
    hosthaloid = np.load(SAM_dir+'GalpropHaloIndex_corr.npy').astype(int)
    mhalo = np.load(SAM_dir+'GalpropMhalo.npy')
    mstar = np.load(SAM_dir+'GalpropMstar.npy')
    sfr = np.load(SAM_dir+'GalpropSfr.npy')
    sat_type = np.load(SAM_dir+'GalpropSatType.npy')

    xyz_position = np.load(SAM_dir+'GalpropPos.npy')
    xyz_position[xyz_position > Lbox] -= Lbox
    xyz_position[xyz_position < 0.] += Lbox 
    
    # halo arrays
    halo_m_vir = np.load(SAM_dir+'HalopropMvir.npy')
    halo_c_nfw = np.load(SAM_dir+'HalopropC_nfw.npy')
    halo_rhalo = np.load(SAM_dir+'GalpropRhalo.npy')[sat_type == 0]
    halo_spin = np.load(SAM_dir+'HalopropSpin.npy')
    halo_sigma_bulge = np.load(SAM_dir+'GalpropSigmaBulge.npy')[sat_type == 0]

    if want_matching_sam or want_matching_hydro:
        # load the matches with hydro
        halo_subfind_id = np.load(SAM_dir+'HalopropSubfindID.npy')
    
N_halos_sam = len(halo_m_vir)

# positions of the halos
halo_xyz_position = xyz_position[sat_type==0]

# environment parameter
density = np.load('../smoothed_mass_in_area.npy')
n_gr = density.shape[0]
print("n_gr = ",n_gr)
gr_size = Lbox/n_gr
halo_ijk = (halo_xyz_position/gr_size).astype(int)
halo_environment = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

##########################################
#####               HYDRO            ##### 
##########################################

# Loading data for the hydro -- you can skip for now
hydro_dir = '/mnt/gosling1/boryanah/TNG100/'
SubGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp.npy')
#GrMcrit_fp = np.load(hydro_dir+'Group_M_Mean200_fp.npy')*1.e10
GrMcrit_fp = np.load(hydro_dir+'Group_M_Crit200_fp.npy')*1.e10
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp.npy')/1000.
SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp.npy')/1000.
SubhaloSFR_fp = np.load(hydro_dir+'SubhaloSFR_fp.npy')
SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp.npy')[:,4]*1.e10
SubhaloID = np.load(hydro_dir+'SubhaloID_fp.npy')
SubMstar_fp[SubhaloID] = np.load(hydro_dir+'SubhaloMstar_30kpc_fp.npy')*h # Msun/h
SubMstarHalf_fp = np.load(hydro_dir+'SubhaloMassInHalfRadType_fp.npy')[:,4]*1.e10


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

if secondary_property == 'env': halo_prop = halo_environment; group_prop = GroupEnv_fp; order_type = 'desc'; sec_label = r'${\rm env. \ (descending)}$'
elif secondary_property == 'rvir': halo_prop = halo_rhalo; group_prop = GrRcrit_fp; order_type = 'desc'; sec_label = r'${\rm vir. \ rad. \ (descending)}$'
elif secondary_property == 'conc': halo_prop = halo_c_nfw; group_prop = (Group_Vmax_fp/Group_V_Crit200_fp); order_type = 'mixed'; sec_label = r'${\rm conc. \ (mixed)}$'
elif secondary_property == 'vdisp': halo_prop = halo_sigma_bulge; group_prop = get_from_sub_prop(SubhaloVelDisp_fp,SubGrNr_fp,N_halos_hydro); order_type = 'mixed'; sec_label = r'${\rm vel. \ disp. \ (mixed)}$'
elif secondary_property == 'spin': halo_prop = halo_spin; group_prop = get_from_sub_prop(SubhaloSpin_fp,SubGrNr_fp,N_halos_hydro); order_type = 'desc'; sec_label = r'${\rm spin \ (descending)}$'
elif secondary_property == 's2r': halo_prop = halo_sigma_bulge**2*halo_rhalo; group_prop = get_from_sub_prop(SubhaloVelDisp_fp**2*SubhaloHalfmassRad_fp,SubGrNr_fp,N_halos_hydro); order_type = 'asc'; sec_label = r'${\rm spin \ (descending)}$'

# --------------------------------------------------------
#                   MATCH SAM HYDRO
# --------------------------------------------------------

if want_matching_sam or want_matching_hydro:
    # obtain matching indices for halos
    hydro_matched, sam_matched = get_match(halo_subfind_id,SubGrNr_fp,N_halos_sam)
    N_halos_matched = len(sam_matched)

    # match halos in FP and SAM
    #GroupPos_fp, halo_xyz_position = match_halos(GroupPos_fp,halo_xyz_position,hydro_matched,sam_matched)


# --------------------------------------------------------
#                         SAM
# --------------------------------------------------------

# order the subhalos in terms of their stellar masses
if type_gal == "mstar":
    inds_top = (np.argsort(mstar)[::-1])[:num_gals_sam] 
elif type_gal == "sfr":
    inds_top = (np.argsort(sfr)[::-1])[:num_gals_sam]
inds_gal = np.arange(len(mstar),dtype=int)
inds_cent = inds_gal[sat_type == 0]
inds_top_cent = np.intersect1d(inds_cent,inds_top)

if want_matching_sam:
    inds_top_cent_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top_cent, hosthaloid, sam_matched, sam_matched)

    if halo_prop is not None:
        
        shmr_sam_top, shmr_sam_bot, bin_cents = get_shmr_prop(inds_top_cent_matched, inds_halo_host_comm1_matched, mstar, halo_m_vir[sam_matched], halo_prop[sam_matched])
        
        plot_shmr_prop(shmr_sam_top, shmr_sam_bot, bin_cents, label='SAM')

        np.save("data_shmr/shmr_sam_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",shmr_sam_top)
        np.save("data_shmr/shmr_sam_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",shmr_sam_bot)
    else:
        
        shmr_sam, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_comm1_matched, mstar, halo_m_vir[sam_matched])
        plot_shmr(shmr_sam, bin_cents, label='hydro')
        np.save("data_shmr/shmr_sam_"+str(num_gals_hydro)+"_"+type_gal+".npy",shmr_sam)
    # versions
    #shmr_sam, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_matched, mstar, halo_m_vir)

else:
    if halo_prop is not None:
        shmr_sam_top, shmr_sam_bot, bin_cents = get_shmr_prop(inds_top_cent, hosthaloid[inds_top_cent], mstar, halo_m_vir, halo_prop)

        plot_shmr_prop(shmr_sam_top, shmr_sam_bot, bin_cents, label='SAM')
        np.save("data_shmr/shmr_sam_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",shmr_sam_top)
        np.save("data_shmr/shmr_sam_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",shmr_sam_bot)
    else:
        shmr_sam, bin_cents = get_shmr(inds_top_cent, hosthaloid[inds_top_cent], mstar, halo_m_vir)
        plot_shmr(shmr_sam, bin_cents, label='SAM')
        np.save("data_shmr/shmr_sam_"+str(num_gals_hydro)+"_"+type_gal+".npy",shmr_sam)



np.save("data_shmr/bin_cents.npy",bin_cents)

# --------------------------------------------------------
#                         HYDRO
# --------------------------------------------------------


# Find the indices of the top galaxies
if type_gal == "mstar":
    inds_top = (np.argsort(SubMstar_fp)[::-1])[:num_gals_hydro] 
elif type_gal == "sfr":
    inds_top = (np.argsort(SubhaloSFR_fp)[::-1])[:num_gals_hydro]
unique_hosts, inds_cent = np.unique(SubGrNr_fp,return_index=True)
inds_top_cent = np.intersect1d(inds_cent,inds_top)

if want_matching_hydro:
    inds_top_cent_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top_cent, SubGrNr_fp, sam_matched, hydro_matched)

    if halo_prop is not None:
        
        shmr_hydro_top, shmr_hydro_bot, bin_cents = get_shmr_prop(inds_top_cent_matched, inds_halo_host_comm1_matched, SubMstar_fp, halo_m_vir[sam_matched], halo_prop[sam_matched])
        
        plot_shmr_prop(shmr_hydro_top, shmr_hydro_bot, bin_cents, label='hydro')

        np.save("data_shmr/shmr_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",shmr_hydro_top)
        np.save("data_shmr/shmr_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",shmr_hydro_bot)
    else:
        
        shmr_hydro, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_comm1_matched, SubMstar_fp, halo_m_vir[sam_matched])
        plot_shmr(shmr_hydro, bin_cents, label='hydro')
        np.save("data_shmr/shmr_hydro_"+str(num_gals_hydro)+"_"+type_gal+".npy",shmr_hydro)
    # versions
    #shmr_hydro, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_matched, SubMstar_fp, halo_m_vir)
    #shmr_hydro, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_comm1_matched, SubMstar_fp, GrMcrit_fp[hydro_matched])
else:

    if halo_prop is not None:
        shmr_hydro_top, shmr_hydro_bot, bin_cents = get_shmr_prop(inds_top_cent,SubGrNr_fp[inds_top_cent], SubMstar_fp, GrMcrit_fp, group_prop)
        
        plot_shmr_prop(shmr_hydro_top, shmr_hydro_bot, bin_cents, label='hydro')

        np.save("data_shmr/shmr_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",shmr_hydro_top)
        np.save("data_shmr/shmr_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",shmr_hydro_bot)
    else:
        # total histogram of sats + centrals and galaxy counts per halo
        shmr_hydro, bin_cents = get_shmr(inds_top_cent, SubGrNr_fp[inds_top_cent], SubMstar_fp, GrMcrit_fp)
        plot_shmr(shmr_hydro, bin_cents, label='hydro')
        np.save("data_shmr/shmr_hydro_"+str(num_gals_hydro)+"_"+type_gal+".npy",shmr_hydro)

plt.legend()
plt.savefig('figs/SHMR.png')
#plt.show()
