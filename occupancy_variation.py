import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotparams # for making plots pretty, but comment out
plotparams.buba() # same
import Corrfunc # comment out since you're not using it
import sys

from tools.matching import get_match, match_halos, match_subs
from tools.halostats import get_from_sub_prop, get_hist_count, get_hod, get_hod_prop, plot_hod, plot_hod_prop
    
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
    #sfr = np.load(SAM_dir+'GalpropSfrave100myr.npy')
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
GrMcrit_fp = np.load(hydro_dir+'Group_M_Crit200_fp.npy')*1.e10
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp.npy')/1000.
SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp.npy')/1000.
SubhaloSFR_fp = np.load(hydro_dir+'SubhaloSFR_fp.npy')
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

if want_matching_sam:
    inds_top_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top, hosthaloid, sam_matched, sam_matched)

    # total histogram of sats + centrals and galaxy counts per halo
    hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, xyz_position, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], record_relative=record_relative)
    
    if halo_prop is not None:
        hist_sam_top, hist_sam_bot, bin_cents = get_hod_prop(halo_m_vir[sam_matched],count_sam,halo_prop[sam_matched])
        plot_hod_prop(hist_sam_top, hist_sam_bot, bin_cents, label='SAM')
        np.save("data_hod/hist_sam_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_sam_top)
        np.save("data_hod/hist_sam_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_sam_bot)
    else:
        hist_sam, bin_cents = get_hod(halo_m_vir[sam_matched],count_sam)
        plot_hod(hist_sam, bin_cents,label='SAM')
        np.save("data_hod/hist_sam_"+str(num_gals_hydro)+"_"+type_gal+".npy",hist_sam)
        
else:
    # total histogram of sats + centrals and galaxy counts per halo
    hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top, hosthaloid[inds_top], xyz_position, halo_m_vir, halo_xyz_position, record_relative=record_relative)

    if halo_prop is not None:
        hist_sam_top, hist_sam_bot, bin_cents = get_hod_prop(halo_m_vir,count_sam,halo_prop)

        plot_hod_prop(hist_sam_top, hist_sam_bot, bin_cents, label='SAM')
        np.save("data_hod/hist_sam_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_sam_top)
        np.save("data_hod/hist_sam_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_sam_bot)
    else:
        hist_sam, bin_cents = get_hod(halo_m_vir,count_sam)
        plot_hod(hist_sam, bin_cents,label='SAM')
        np.save("data_hod/hist_sam_"+str(num_gals_hydro)+"_"+type_gal+".npy",hist_sam)

np.save("data_hod/bin_cents.npy",bin_cents)

# --------------------------------------------------------
#                         HYDRO
# --------------------------------------------------------


# Find the indices of the top galaxies
if type_gal == "mstar":
    inds_top = np.argsort(SubMstar_fp)[::-1][:num_gals_hydro]
elif type_gal == "sfr":
    inds_top = np.argsort(SubhaloSFR_fp)[::-1][:num_gals_hydro]

if want_matching_hydro:
    inds_top_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top, SubGrNr_fp, sam_matched, hydro_matched)

    # total histogram of sats + centrals and galaxy counts per halo
    hist_hydro, bin_cents, count_fp, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, SubhaloPos_fp, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], record_relative=record_relative)

    # version using hydro halos (change down [hydro_matched])
    #hist_hydro, bin_cents, count_fp, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_matched, SubhaloPos_fp, GrMcrit_fp, GroupPos_fp, record_relative=record_relative)

    
    if halo_prop is not None:
        
        hist_hydro_top, hist_hydro_bot, bin_cents = get_hod_prop(halo_m_vir[sam_matched],count_fp,halo_prop[sam_matched])
        plot_hod_prop(hist_hydro_top, hist_hydro_bot, bin_cents, label='hydro')

        np.save("data_hod/hist_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_hydro_top)
        np.save("data_hod/hist_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_hydro_bot)
    else:
        
        hist_hydro, bin_cents = get_hod(halo_m_vir[sam_matched],count_fp)
        plot_hod(hist_hydro, bin_cents, label='hydro')
        np.save("data_hod/hist_hydro_"+str(num_gals_hydro)+"_"+type_gal+".npy",hist_hydro)

else:
    # total histogram of sats + centrals and galaxy counts per halo
    hist_hydro, bin_cents, count_fp, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top, SubGrNr_fp[inds_top], SubhaloPos_fp, GrMcrit_fp, GroupPos_fp, record_relative=record_relative)

    if halo_prop is not None:
        hist_hydro_top, hist_hydro_bot, bin_cents = get_hod_prop(GrMcrit_fp,count_fp,group_prop)
        plot_hod_prop(hist_hydro_top, hist_hydro_bot, bin_cents, label='hydro')

        np.save("data_hod/hist_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_hydro_top)
        np.save("data_hod/hist_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_hydro_bot)
    else:
        hist_hydro, bin_cents = get_hod(GrMcrit_fp,count_fp)
        plot_hod(hist_hydro, bin_cents, label='hydro')
        np.save("data_hod/hist_hydro_"+str(num_gals_hydro)+"_"+type_gal+".npy",hist_hydro)
        
plt.legend()
plt.savefig('figs/HOD_'+secondary_property+'_'+type_gal+'.png')
#plt.show()
