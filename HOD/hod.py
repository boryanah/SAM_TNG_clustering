#!/usr/bin/env python3
'''
Script for computing and saving HOD.

Usage:
------
./hod.py --help
'''

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

import plotparams # for making plots pretty, but comment out
plotparams.buba() # same
from tools.matching import get_match, match_halos, match_subs
from tools.halostats import get_from_sub_prop, get_hist_count, get_hod, get_hod_prop, plot_hod, plot_hod_prop

DEFAULTS = {}
DEFAULTS['h'] = 0.6774
DEFAULTS['snapshot'] = 99
DEFAULTS['hydro_dir'] = '/mnt/gosling1/boryanah/TNG100/'
DEFAULTS['SAM_dir'] = '/mnt/store1/boryanah/SAM_subvolumes/'
DEFAULTS['num_gals'] = 6000
DEFAULTS['Lbox'] = 75.
DEFAULTS['type_gal'] = "mstar" #"mstar"#"sfr"#"mhalo"
DEFAULTS['secondary_property'] = 'shuff'

def main(type_gal, secondary_property, num_gals, want_matching_sam=False, want_matching_hydro=False, record_relative=True, h=DEFAULTS['h'], snapshot=DEFAULTS['snapshot'], hydro_dir=DEFAULTS['hydro_dir'], SAM_dir=DEFAULTS['SAM_dir'], Lbox=DEFAULTS['Lbox']):
    # some default parameters
    num_gals_hydro = num_gals
    num_gals_sam = num_gals
    n_gal = num_gals_sam/Lbox**3.

    if secondary_property == 'shuff':
        halo_prop = None
        group_prop = None
        order_type = 'mixed' # doesn't matter
        sec_label = 'shuffled'
    
    ##########################################
    #####               SAM              ##### 
    ##########################################

    # subhalo arrays
    hosthaloid = np.load(SAM_dir+'GalpropHaloIndex.npy').astype(int)
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
    density = np.load(hydro_dir+'smoothed_mass_in_area.npy')
    n_gr = density.shape[0]
    print("n_gr = ",n_gr)
    gr_size = Lbox/n_gr
    halo_ijk = (halo_xyz_position/gr_size).astype(int)
    halo_environment = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

    ##########################################
    #####               HYDRO            ##### 
    ##########################################

    # Loading data for the hydro -- you can skip for now
    SubGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp.npy')
    GrMcrit_fp = np.load(hydro_dir+'Group_M_Crit200_fp.npy')*1.e10
    GroupPos_fp = np.load(hydro_dir+'GroupPos_fp.npy')/1000.
    SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp.npy')/1000.
    SubhaloSFR_fp = np.load(hydro_dir+'SubhaloSFR_fp.npy')
    # TESTING
    SubMstar_fp = np.load(hydro_dir+'SubhaloMassInRadType_fp.npy')[:,4]*1.e10
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassInHalfRadType_fp.npy')[:,4]*1.e10
    # og
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp.npy')[:,4]*1.e10



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
    elif type_gal == "mhalo":
        halo_mass_matched = mhalo[sat_type == 0][sam_matched]
        inds_halo_top_matched = (np.argsort(halo_mass_matched)[::-1])[:num_gals_sam]
        bool_arr = np.in1d(hosthaloid, sam_matched[inds_halo_top_matched])
        mstar_low = 1.e10
        bool_arr = (mstar > mstar_low) & bool_arr
        inds_top = np.arange(len(mstar))[bool_arr]
        print(len(inds_top))
        #inds_top = (np.argsort(mhalo)[::-1])[:num_gals_sam]
        mhalo_low = (np.sort(mhalo)[::-1])[num_gals_sam]
        print("minimum halo mass = ",mhalo_low)
    elif type_gal == "mhalo_cent":
        halo_mass_matched = mhalo[sat_type == 0][sam_matched]
        inds_halo_top_matched = (np.argsort(halo_mass_matched)[::-1])[:num_gals_sam]
        bool_arr = np.in1d(hosthaloid, sam_matched[inds_halo_top_matched])
        bool_arr = (mstar > mstar_low) & bool_arr
        inds_top = np.arange(len(mstar))[bool_arr]
        #inds_top = (np.argsort(mhalo)[::-1])[:num_gals_sam]
        mhalo_low = (np.sort(mhalo)[::-1])[num_gals_sam]
        print("minimum halo mass = %.3e"%mhalo_low)

        inds_gal = np.arange(len(mstar),dtype=int)
        inds_cent = inds_gal[sat_type == 0]
        inds_top = np.intersect1d(inds_cent,inds_top)
        print(len(inds_top))

    if want_matching_sam:
        inds_top_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top, hosthaloid, sam_matched, sam_matched)

        # total histogram of sats + centrals and galaxy counts per halo
        hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, xyz_position, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], record_relative=record_relative)

        if halo_prop is not None:
            hist_sam_top, hist_sam_bot, bin_cents = get_hod_prop(halo_m_vir[sam_matched],count_sam,halo_prop[sam_matched])
            plot_hod_prop(hist_sam_top, hist_sam_bot, bin_cents, label='SAM')
            np.save("data/hist_sam_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_sam_top)
            np.save("data/hist_sam_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_sam_bot)
        else:
            hist_sam, bin_cents = get_hod(halo_m_vir[sam_matched],count_sam)
            plot_hod(hist_sam, bin_cents,label='SAM')
            np.save("data/hist_sam_"+str(num_gals_hydro)+"_"+type_gal+".npy",hist_sam)

    else:
        # total histogram of sats + centrals and galaxy counts per halo
        hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top, hosthaloid[inds_top], xyz_position, halo_m_vir, halo_xyz_position, record_relative=record_relative)

        if halo_prop is not None:
            hist_sam_top, hist_sam_bot, bin_cents = get_hod_prop(halo_m_vir,count_sam,halo_prop)

            plot_hod_prop(hist_sam_top, hist_sam_bot, bin_cents, label='SAM')
            np.save("data/hist_sam_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_sam_top)
            np.save("data/hist_sam_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_sam_bot)
        else:
            hist_sam, bin_cents = get_hod(halo_m_vir,count_sam)
            plot_hod(hist_sam, bin_cents,label='SAM')
            np.save("data/hist_sam_"+str(num_gals_hydro)+"_"+type_gal+".npy",hist_sam)

    np.save("data/bin_cents.npy",bin_cents)

    # --------------------------------------------------------
    #                         HYDRO
    # --------------------------------------------------------


    # Find the indices of the top galaxies
    if type_gal == "mstar":
        inds_top = np.argsort(SubMstar_fp)[::-1][:num_gals_hydro]
    elif type_gal == "sfr":
        inds_top = np.argsort(SubhaloSFR_fp)[::-1][:num_gals_hydro]
    elif type_gal == "mhalo":
        #halo_mass_matched = mhalo[sat_type == 0][sam_matched] # GrMcrit_fp
        #inds_halo_top_matched = (np.argsort(halo_mass_matched)[::-1])[:num_gals_sam]
        bool_arr = np.in1d(SubGrNr_fp, hydro_matched[inds_halo_top_matched])
        bool_arr = (SubMstar_fp > mstar_low) & bool_arr
        inds_top = np.arange(len(SubMstar_fp))[bool_arr]
        print(len(inds_top))
    elif type_gal == "mhalo_cent":
        #halo_mass_matched = mhalo[sat_type == 0][sam_matched] # GrMcrit_fp
        #inds_halo_top_matched = (np.argsort(halo_mass_matched)[::-1])[:num_gals_sam]
        bool_arr = np.in1d(SubGrNr_fp, hydro_matched[inds_halo_top_matched])
        bool_arr = (SubMstar_fp > mstar_low) & bool_arr
        inds_top = np.arange(len(SubMstar_fp))[bool_arr]

        unique_hosts, inds_cent = np.unique(SubGrNr_fp,return_index=True)
        inds_top = np.intersect1d(inds_cent,inds_top)
        print(len(inds_top))

    if want_matching_hydro:
        inds_top_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top, SubGrNr_fp, sam_matched, hydro_matched)

        # total histogram of sats + centrals and galaxy counts per halo
        hist_hydro, bin_cents, count_fp, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, SubhaloPos_fp, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], record_relative=record_relative)

        # version using hydro halos (change down [hydro_matched])
        #hist_hydro, bin_cents, count_fp, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_matched, SubhaloPos_fp, GrMcrit_fp, GroupPos_fp, record_relative=record_relative)


        if halo_prop is not None:

            hist_hydro_top, hist_hydro_bot, bin_cents = get_hod_prop(halo_m_vir[sam_matched],count_fp,halo_prop[sam_matched])
            plot_hod_prop(hist_hydro_top, hist_hydro_bot, bin_cents, label='hydro')

            np.save("data/hist_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_hydro_top)
            np.save("data/hist_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_hydro_bot)
        else:

            hist_hydro, bin_cents = get_hod(halo_m_vir[sam_matched],count_fp)
            plot_hod(hist_hydro, bin_cents, label='hydro')
            np.save("data/hist_hydro_"+str(num_gals_hydro)+"_"+type_gal+".npy",hist_hydro)

    else:
        # total histogram of sats + centrals and galaxy counts per halo
        hist_hydro, bin_cents, count_fp, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top, SubGrNr_fp[inds_top], SubhaloPos_fp, GrMcrit_fp, GroupPos_fp, record_relative=record_relative)

        if halo_prop is not None:
            hist_hydro_top, hist_hydro_bot, bin_cents = get_hod_prop(GrMcrit_fp,count_fp,group_prop)
            plot_hod_prop(hist_hydro_top, hist_hydro_bot, bin_cents, label='hydro')

            np.save("data/hist_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_hydro_top)
            np.save("data/hist_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+".npy",hist_hydro_bot)
        else:
            hist_hydro, bin_cents = get_hod(GrMcrit_fp,count_fp)
            plot_hod(hist_hydro, bin_cents, label='hydro')
            np.save("data/hist_hydro_"+str(num_gals_hydro)+"_"+type_gal+".npy",hist_hydro)

    plt.legend()
    plt.savefig('figs/HOD_'+secondary_property+'_'+type_gal+'.png')
    #plt.show()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--type_gal', help='Galaxy type', default=DEFAULTS['type_gal'])
    parser.add_argument('--secondary_property', help='Secondary property', default=DEFAULTS['secondary_property'])
    parser.add_argument('--num_gals', help='Number of galaxies', type=int, default=DEFAULTS['num_gals'])
    parser.add_argument('--want_matching_sam', help='Match SAM?', action='store_true')
    parser.add_argument('--want_matching_hydro', help='Match Hydro?', action='store_true')
    args = parser.parse_args()
    args = vars(args)
    main(**args)
