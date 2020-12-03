import matplotlib.pyplot as plt
import numpy as np
import plotparams # for making plots pretty, but comment out
plotparams.buba() # same
import Corrfunc # comment out since you're not using it
import sys

from tools.matching import get_match, match_halos, match_subs
from tools.halostats import get_hist_count, get_xyz_w, get_jack_corr, get_shuff_counts, get_from_sub_prop

np.random.seed(300)

# Hubble constant, snapshot number (99 is last snapshot, i.e. z = 0), number of galaxies
h = 0.6774
k = 99
num_gals_hydro = int(sys.argv[1])
rough_percent_matches = 98.6
num_gals_sam = int(rough_percent_matches/100*num_gals_hydro)
Lbox = 75.
n_gal = num_gals_sam/Lbox**3.
want_matching_sam = True#False
want_matching_hydro = True
record_relative = 1
type_gal = sys.argv[2]#"sfr"#"mstar"
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

# choose only the host halo positions
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
elif secondary_property == 'conc_desc': halo_prop = halo_c_nfw; group_prop = (Group_Vmax_fp/Group_V_Crit200_fp); order_type = 'desc'; sec_label = r'${\rm conc. \ (descending)}$'
elif secondary_property == 'conc_asc': halo_prop = halo_c_nfw; group_prop = (Group_Vmax_fp/Group_V_Crit200_fp); order_type = 'asc'; sec_label = r'${\rm conc. \ (ascending)}$'
elif secondary_property == 'vdisp': halo_prop = halo_sigma_bulge; group_prop = get_from_sub_prop(SubhaloVelDisp_fp,SubGrNr_fp,N_halos_hydro); order_type = 'mixed'; sec_label = r'${\rm vel. \ disp. \ (mixed)}$'
elif secondary_property == 'vdisp_desc': halo_prop = halo_sigma_bulge; group_prop = get_from_sub_prop(SubhaloVelDisp_fp,SubGrNr_fp,N_halos_hydro); order_type = 'desc'; sec_label = r'${\rm vel. \ disp. \ (descending)}$'
elif secondary_property == 'vdisp_asc': halo_prop = halo_sigma_bulge; group_prop = get_from_sub_prop(SubhaloVelDisp_fp,SubGrNr_fp,N_halos_hydro); order_type = 'asc'; sec_label = r'${\rm vel. \ disp. \ (ascending)}$'
elif secondary_property == 'spin_asc': halo_prop = halo_spin; group_prop = get_from_sub_prop(SubhaloSpin_fp,SubGrNr_fp,N_halos_hydro); order_type = 'asc'; sec_label = r'${\rm spin \ (descending)}$'
elif secondary_property == 'spin_desc': halo_prop = halo_spin; group_prop = get_from_sub_prop(SubhaloSpin_fp,SubGrNr_fp,N_halos_hydro); order_type = 'desc'; sec_label = r'${\rm spin \ (descending)}$'
elif secondary_property == 's2r_asc': halo_prop = halo_sigma_bulge**2*halo_rhalo; group_prop = get_from_sub_prop(SubhaloVelDisp_fp**2*SubhaloHalfmassRad_fp,SubGrNr_fp,N_halos_hydro); order_type = 'asc'; sec_label = r'${\rm spin \ (descending)}$'
elif secondary_property == 's2r_desc': halo_prop = halo_sigma_bulge**2*halo_rhalo; group_prop = get_from_sub_prop(SubhaloVelDisp_fp**2*SubhaloHalfmassRad_fp,SubGrNr_fp,N_halos_hydro); order_type = 'desc'; sec_label = r'${\rm spin \ (descending)}$'


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

    print(len(inds_top_matched))
    # limit to only matched
    # TESTING
    #inds_top_matched, inds_halo_host_comm1_matched = inds_top_matched[:8000], inds_halo_host_comm1_matched[:8000]
    
    # total histogram of sats + centrals and galaxy counts per halo
    hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, xyz_position, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], record_relative=record_relative)

    # shuffle halo occupations
    count_nstart_sam = np.vstack((count_sam,nstart_sam)).T
    if halo_prop is not None:
        count_hod, nstart_hod = get_shuff_counts(count_nstart_sam,halo_m_vir[sam_matched],record_relative=record_relative,order_by=halo_prop[sam_matched],order_type=order_type)
    else:
        count_hod, nstart_hod = get_shuff_counts(count_nstart_sam,halo_m_vir[sam_matched],record_relative=record_relative,order_by=halo_prop,order_type=order_type)

else:
    # total histogram of sats + centrals and galaxy counts per halo
    hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top, hosthaloid[inds_top], xyz_position, halo_m_vir, halo_xyz_position, record_relative=record_relative)

    # version 1 -- random draws for satellites and centrals
    # get the centrals counts
    #count_cents_sam = count_sam.copy()
    # assume that every halo with at least one galaxy has a central
    #count_cents_sam[count_sam > 0] = 1
    #hist_sam, bin_cents = get_hod(halo_m_vir,count_sam,halo_m_vir)
    #hist_cents_sam, bin_cents = get_hod(halo_m_vir,count_cents_sam,halo_m_vir)
    #hist_sats_sam = hist_sam - hist_cents_sam
    # random draws (has to pick a strategy for small scales; for now just centers)
    #count_hod = get_hod_counts(hist_sam,hist_cents_sam,hist_sats_sam,N_halos_sam,halo_m_vir)
    # version 2 -- putting relative to centers
    count_nstart_sam = np.vstack((count_sam,nstart_sam)).T
    count_hod, nstart_hod = get_shuff_counts(count_nstart_sam,halo_m_vir,record_relative=record_relative,order_by=halo_prop,order_type=order_type)
    
# our halo positions are ordered in terms of stellar mass so simply select the top 6000 -> those are our true positions of the galaxies
xyz_true = xyz_position[inds_top]
w_true = np.ones(xyz_true.shape[0],dtype=xyz_true.dtype)


# here we select out of our newly constructed array with counts per halo only those halos which have galaxies and put the galaxies in the center (weights are not all ones)
#xyz_cents = halo_xyz_position[count_sam>0]
#w_cents = count_sam[count_sam>0].astype(xyz_cents.dtype)

if want_matching_sam:
    xyz_hod, w_hod = get_xyz_w(count_hod,nstart_hod,halo_xyz_position[sam_matched],rel_pos_gals_sam,xyz_true.dtype,Lbox)
else:
    # version 1 -- put in center
    # this is the shuffled HOD where again we put the galaxies at the center of the halos
    #xyz_hod = halo_xyz_position[count_hod > 0]
    #w_hod = count_hod[count_hod > 0].astype(xyz_hod.dtype)
    # version 2 -- put relative positions

    xyz_hod, w_hod = get_xyz_w(count_hod,nstart_hod,halo_xyz_position,rel_pos_gals_sam,xyz_true.dtype,Lbox)


# Correlation function ratios for the SAM

'''
# compute the ratio of the corr func of the constructed HOD to the true SAM with error bars and plot it
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_cents,w_cents,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='--',label='SAM: halo cents/true',alpha=1.,fmt='o',capsize=4)
'''

# compute the ratio of the shuffled galaxies to the true SAM with error bars and plot it
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)

plt.figure(1,figsize=(8,7))

line = np.linspace(0,40,3)
plt.plot(line,np.ones(len(line)),'k--')

plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='-',label='SAM',alpha=1.,fmt='o',capsize=4)

np.save("data_rat/rat_mean_sam_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",Rat_hodtrue_mean)
np.save("data_rat/rat_err_sam_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",Rat_hodtrue_err)

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

    # limit to only matched
    # TESTING
    #inds_top_matched, inds_halo_host_comm1_matched = inds_top_matched[:5000], inds_halo_host_comm1_matched[:5000]
    
    # total histogram of sats + centrals and galaxy counts per halo
    hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, SubhaloPos_fp, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], record_relative=record_relative)
    # using hydro centers
    #hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, SubhaloPos_fp, halo_m_vir[sam_matched], GroupPos_fp[hydro_matched], record_relative=record_relative)

    count_nstart_fp = np.vstack((count_fp,nstart_fp)).T
    if halo_prop is not None:
        count_hod_fp, nstart_hod_fp = get_shuff_counts(count_nstart_fp,halo_m_vir[sam_matched],record_relative=record_relative,order_by=halo_prop[sam_matched],order_type=order_type)
    else:
        count_hod_fp, nstart_hod_fp = get_shuff_counts(count_nstart_fp,halo_m_vir[sam_matched],record_relative=record_relative,order_by=halo_prop,order_type=order_type)
else:

    # total histogram of sats + centrals and galaxy counts per halo
    hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top,SubGrNr_fp[inds_top], SubhaloPos_fp, GrMcrit_fp, GroupPos_fp,  record_relative=record_relative)

    # version 1
    # get parent indices of the centrals and their subhalo indices in the original array
    #unique_sub_grnr, firsts = np.unique(SubGrNr_fp,return_index=True)
    #inds_firsts_gals = np.intersect1d(inds_top,firsts)
    # central histogram
    #hist_hydro_cents, bin_cents, count_halo_sat_fp = get_hist_count(inds_firsts_gals)
    #hist_hydro_sats = hist_hydro-hist_hydro_cents
    #count_hod_fp = get_hod_counts(hist_hydro,hist_hydro_cents,hist_hydro_sats,N_halos_hydro,GrMcrit_fp)
    # version 2 -- putting in centers
    count_nstart_fp = np.vstack((count_fp,nstart_fp)).T
    count_hod_fp, nstart_hod_fp = get_shuff_counts(count_nstart_fp,GrMcrit_fp,record_relative=record_relative,order_by=group_prop,order_type=order_type)
    
# select the galaxies positions and weights
xyz_true = SubhaloPos_fp[inds_top]
w_true = np.ones(xyz_true.shape[0],dtype=xyz_true.dtype)
    
# here we select out of our newly constructed array with counts per halo only those halos which have galaxies and put the galaxies in the center
#xyz_cents = GroupPos_fp[count_fp > 0]
#w_cents = count_fp[count_fp > 0].astype(xyz_cents.dtype)


# version 1 -- put in centers of halos
# this is the shuffled HOD where again we put the galaxies at the center of the halos
#xyz_hod = GroupPos_fp[count_hod_fp > 0]
#w_hod = count_hod_fp[count_hod_fp > 0].astype(xyz_hod.dtype)
# version 2 -- put relative positions
if want_matching_hydro:
    xyz_hod, w_hod = get_xyz_w(count_hod_fp,nstart_hod_fp,halo_xyz_position[sam_matched],rel_pos_gals_fp,xyz_true.dtype,Lbox)
else:
    xyz_hod, w_hod = get_xyz_w(count_hod_fp,nstart_hod_fp,GroupPos_fp,rel_pos_gals_fp,xyz_true.dtype,Lbox)

# Correlation function ratios for the hydro

'''
# compute the ratio of the corr func of the constructed HOD to the true hydro with error bars and plot it
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_cents,w_cents,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='orange',ls='--',label='Hydro: halo cents/true',alpha=1.,fmt='o',capsize=4)
'''

# compute the ratio of the shuffled galaxies to the true hydro with error bars and plot it
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)

plt.figure(1,figsize=(8,7))

plt.errorbar(bin_centers*1.05,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='orange',ls='-',label='Hydro',alpha=1.,fmt='o',capsize=4)

np.save("data_rat/rat_mean_hydro_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",Rat_hodtrue_mean)
np.save("data_rat/rat_err_hydro_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",Rat_hodtrue_err)

np.save("data_rat/bin_centers.npy",bin_centers)

plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.xlim([0.1,13])
plt.ylim([0.4,1.5])
plt.text(0.2,0.5,sec_label)
plt.savefig("figs/mock_ratio_"+secondary_property+".png")
