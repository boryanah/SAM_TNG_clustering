#!/usr/bin/env python3
'''
Script for computing and saving SHMR.

Usage:
------
./shmr.py --help
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
from tools.halostats import get_from_sub_prop, get_shmr, get_shmr_prop, plot_shmr, plot_shmr_prop

DEFAULTS = {}
DEFAULTS['h'] = 0.6774
DEFAULTS['snapshot'] = 99
DEFAULTS['Lbox'] = 205.
if np.abs(DEFAULTS['Lbox'] - 75.) < 1.e-6: sim_name = 'TNG100'
elif np.abs(DEFAULTS['Lbox'] - 205.) < 1.e-6: sim_name = 'TNG300'
DEFAULTS['hydro_dir'] = '/mnt/gosling1/boryanah/'+sim_name+'/'
DEFAULTS['SAM_dir'] = '/mnt/store1/boryanah/SAM_subvolumes_'+sim_name+'/'
DEFAULTS['num_gals'] = 12000
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
        order_type = 'mixed' # whatever
        sec_label = 'shuffled'

    if snapshot == 99:
        str_snap = ''
    else:
        str_snap = '_%d'%snapshot
        
    ##########################################
    #####               SAM              ##### 
    ##########################################

    # Loading data for the SAM

    # subhalo arrays
    hosthaloid = np.load(SAM_dir+'GalpropHaloIndex'+str_snap+'.npy').astype(int)
    mhalo = np.load(SAM_dir+'GalpropMhalo'+str_snap+'.npy')
    mstar = np.load(SAM_dir+'GalpropMstar'+str_snap+'.npy')
    sfr = np.load(SAM_dir+'GalpropSfr'+str_snap+'.npy')
    sfr /= mstar
    sat_type = np.load(SAM_dir+'GalpropSatType'+str_snap+'.npy')

    xyz_position = np.load(SAM_dir+'GalpropPos'+str_snap+'.npy')
    xyz_position[xyz_position > Lbox] -= Lbox
    xyz_position[xyz_position < 0.] += Lbox 

    # halo arrays
    halo_m_vir = np.load(SAM_dir+'HalopropMvir'+str_snap+'.npy')
    halo_c_nfw = np.load(SAM_dir+'HalopropC_nfw'+str_snap+'.npy')
    halo_rhalo = np.load(SAM_dir+'GalpropRhalo'+str_snap+'.npy')[sat_type == 0]
    halo_mpeak = np.load(SAM_dir+'HalopropMvir_peak'+str_snap+'.npy')
    #halo_tform = np.load(SAM_dir+'GalpropTsat'+str_snap+'.npy')[sat_type == 0]
    #halo_tform = np.load(SAM_dir+'GalpropTmerger'+str_snap+'.npy')[sat_type == 0]
    # small z corresponds to large t
    halo_tform = np.load(SAM_dir+'Halopropz_Mvir_half'+str_snap+'.npy')[::-1]
    halo_vdiskpeak = np.load(SAM_dir+'HalopropVdisk_peak'+str_snap+'.npy')
    halo_spin = np.load(SAM_dir+'HalopropSpin'+str_snap+'.npy')
    halo_sigma_bulge = np.load(SAM_dir+'GalpropSigmaBulge'+str_snap+'.npy')[sat_type == 0]

    if want_matching_sam or want_matching_hydro:
        # load the matches with hydro
        halo_subfind_id = np.load(SAM_dir+'HalopropSubfindID'+str_snap+'.npy')

    N_halos_sam = len(halo_m_vir)

    # positions of the halos
    halo_xyz_position = xyz_position[sat_type==0]

    # environment parameter
    density = np.load(hydro_dir+'smoothed_mass_in_area'+str_snap+'.npy')
    n_gr = density.shape[0]
    print("n_gr = ",n_gr)
    gr_size = Lbox/n_gr
    halo_ijk = (halo_xyz_position/gr_size).astype(int)
    halo_environment = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

    ##########################################
    #####               HYDRO            ##### 
    ##########################################

    # Loading data for the hydro -- you can skip for now
    SubGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp'+str_snap+'.npy')
    #GrMcrit_fp = np.load(hydro_dir+'Group_M_Mean200_fp'+str_snap+'.npy')*1.e10
    #GrMcrit_fp = np.load(hydro_dir+'Group_M_Crit200_fp'+str_snap+'.npy')*1.e10
    GrMcrit_fp = np.load(hydro_dir+'Group_M_TopHat200_fp'+str_snap+'.npy')*1.e10
    GroupPos_fp = np.load(hydro_dir+'GroupPos_fp'+str_snap+'.npy')/1000.
    SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp'+str_snap+'.npy')/1000.
    # this might be best
    SubMstar_fp = np.load(hydro_dir+'SubhaloMassInRadType_fp'+str_snap+'.npy')[:,4]*1.e10
    # TESTING
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassInHalfRadType_fp'+str_snap+'.npy')[:,4]*1.e10
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassInMaxRadType_fp'+str_snap+'.npy')[:,4]*1.e10
    # og
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp'+str_snap+'.npy')[:,4]*1.e10
    SubhaloSFR_fp = np.load(hydro_dir+'SubhaloSFR_fp'+str_snap+'.npy')
    SubhaloSFR_fp /= SubMstar_fp
    #SubhaloID = np.load(hydro_dir+'SubhaloID_fp'+str_snap+'.npy')
    #SubMstar_fp[SubhaloID] = np.load(hydro_dir+'SubhaloMstar_30kpc_fp'+str_snap+'.npy')*h # Msun/h


    # total number of halos
    N_halos_hydro = len(GrMcrit_fp)

    # environment
    halo_ijk = (GroupPos_fp/gr_size).astype(int)%n_gr
    GroupEnv_fp = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

    #GrRcrit_fp = np.load(hydro_dir+'Group_R_Crit200_fp'+str_snap+'.npy')
    GrRcrit_fp = np.load(hydro_dir+'Group_R_TopHat200_fp'+str_snap+'.npy')
    Group_Vmax_fp = np.load(hydro_dir+'Group_Vmax_fp'+str_snap+'.npy')/1000.
    #Group_R_Mean200_fp = np.load(hydro_dir+'Group_R_Mean200_fp'+str_snap+'.npy')/1000
    #Group_V_Crit200_fp = np.load(hydro_dir+'Group_V_Crit200_fp'+str_snap+'.npy')/1000
    Group_V_Crit200_fp = np.load(hydro_dir+'Group_V_TopHat200_fp'+str_snap+'.npy')/1000
    SubhaloSpin_fp = np.sqrt(np.sum(np.load(hydro_dir+'SubhaloSpin_fp'+str_snap+'.npy')**2,axis=1))
    SubhaloVelDisp_fp = np.load(hydro_dir+'SubhaloVelDisp_fp'+str_snap+'.npy')
    SubhaloHalfmassRad_fp = np.load(hydro_dir+'SubhaloHalfmassRad_fp'+str_snap+'.npy')

    if secondary_property == 'env': halo_prop = halo_environment; group_prop = GroupEnv_fp; order_type = 'desc'; sec_label = r'${\rm env. \ (descending)}$'
    elif secondary_property == 'rvir': halo_prop = halo_rhalo; group_prop = GrRcrit_fp; order_type = 'desc'; sec_label = r'${\rm vir. \ rad. \ (descending)}$'
    elif secondary_property == 'mpeak': halo_prop = halo_mpeak; group_prop = np.zeros(len(GrRcrit_fp)); order_type = 'desc'; sec_label = r'${\rm peak mass \ (descending)}$'
    elif secondary_property == 'vdiskpeak': halo_prop = halo_vdiskpeak; group_prop = np.zeros(len(GrRcrit_fp)); order_type = 'desc'; sec_label = r'${\rm peak disk vel. \ (descending)}$'
    elif secondary_property == 'tform': halo_prop = halo_tform; group_prop = np.zeros(len(GrRcrit_fp)); order_type = 'desc'; sec_label = r'${\rm formation time \ (descending)}$'
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

    inds_gal = np.arange(len(mstar),dtype=int)
    inds_cent = inds_gal[sat_type == 0]
    inds_top_cent = np.intersect1d(inds_cent,inds_top)

    if want_matching_sam:
        inds_top_cent_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top_cent, hosthaloid, sam_matched, sam_matched)

        if halo_prop is not None:

            shmr_sam_top, shmr_sam_bot, bin_cents = get_shmr_prop(inds_top_cent_matched, inds_halo_host_comm1_matched, mstar, halo_m_vir[sam_matched], halo_prop[sam_matched])

            plot_shmr_prop(shmr_sam_top, shmr_sam_bot, bin_cents, label='SAM')

            np.save("data/shmr_sam_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+''+str_snap+'.npy',shmr_sam_top)
            np.save("data/shmr_sam_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+''+str_snap+'.npy',shmr_sam_bot)
        else:

            shmr_sam, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_comm1_matched, mstar, halo_m_vir[sam_matched])
            plot_shmr(shmr_sam, bin_cents, label='hydro')
            np.save("data/shmr_sam_"+str(num_gals_hydro)+"_"+type_gal+''+str_snap+'.npy',shmr_sam)
        # versions
        #shmr_sam, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_matched, mstar, halo_m_vir)

    else:
        if halo_prop is not None:
            shmr_sam_top, shmr_sam_bot, bin_cents = get_shmr_prop(inds_top_cent, hosthaloid[inds_top_cent], mstar, halo_m_vir, halo_prop)

            plot_shmr_prop(shmr_sam_top, shmr_sam_bot, bin_cents, label='SAM')
            np.save("data/shmr_sam_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+''+str_snap+'.npy',shmr_sam_top)
            np.save("data/shmr_sam_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+''+str_snap+'.npy',shmr_sam_bot)
        else:
            shmr_sam, bin_cents = get_shmr(inds_top_cent, hosthaloid[inds_top_cent], mstar, halo_m_vir)
            plot_shmr(shmr_sam, bin_cents, label='SAM')
            np.save("data/shmr_sam_"+str(num_gals_hydro)+"_"+type_gal+''+str_snap+'.npy',shmr_sam)



    np.save("data/bin_cents.npy",bin_cents)

    # --------------------------------------------------------
    #                         HYDRO
    # --------------------------------------------------------


    # Find the indices of the top galaxies
    if type_gal == "mstar":
        inds_top = (np.argsort(SubMstar_fp)[::-1])[:num_gals_hydro] 
    elif type_gal == "sfr":
        inds_top = (np.argsort(SubhaloSFR_fp)[::-1])[:num_gals_hydro]
    elif type_gal == "mhalo":
        #halo_mass_matched = mhalo[sat_type == 0][sam_matched] # GrMcrit_fp
        #inds_halo_top_matched = (np.argsort(halo_mass_matched)[::-1])[:num_gals_sam]
        bool_arr = np.in1d(SubGrNr_fp, hydro_matched[inds_halo_top_matched])
        bool_arr = (SubMstar_fp > mstar_low) & bool_arr
        inds_top = np.arange(len(SubMstar_fp))[bool_arr]
        print(len(inds_top))
    unique_hosts, inds_cent = np.unique(SubGrNr_fp,return_index=True)
    inds_top_cent = np.intersect1d(inds_cent,inds_top)

    if want_matching_hydro:
        inds_top_cent_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top_cent, SubGrNr_fp, sam_matched, hydro_matched)

        if halo_prop is not None:

            shmr_hydro_top, shmr_hydro_bot, bin_cents = get_shmr_prop(inds_top_cent_matched, inds_halo_host_comm1_matched, SubMstar_fp, halo_m_vir[sam_matched], halo_prop[sam_matched])

            plot_shmr_prop(shmr_hydro_top, shmr_hydro_bot, bin_cents, label='hydro')

            np.save("data/shmr_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+''+str_snap+'.npy',shmr_hydro_top)
            np.save("data/shmr_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+''+str_snap+'.npy',shmr_hydro_bot)
        else:

            shmr_hydro, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_comm1_matched, SubMstar_fp, halo_m_vir[sam_matched])
            plot_shmr(shmr_hydro, bin_cents, label='hydro')
            np.save("data/shmr_hydro_"+str(num_gals_hydro)+"_"+type_gal+''+str_snap+'.npy',shmr_hydro)
        # versions
        #shmr_hydro, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_matched, SubMstar_fp, halo_m_vir)
        #shmr_hydro, bin_cents = get_shmr(inds_top_cent_matched, inds_halo_host_comm1_matched, SubMstar_fp, GrMcrit_fp[hydro_matched])
    else:

        if halo_prop is not None:
            shmr_hydro_top, shmr_hydro_bot, bin_cents = get_shmr_prop(inds_top_cent,SubGrNr_fp[inds_top_cent], SubMstar_fp, GrMcrit_fp, group_prop)

            plot_shmr_prop(shmr_hydro_top, shmr_hydro_bot, bin_cents, label='hydro')

            np.save("data/shmr_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+''+str_snap+'.npy',shmr_hydro_top)
            np.save("data/shmr_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+"_"+secondary_property+''+str_snap+'.npy',shmr_hydro_bot)
        else:
            # total histogram of sats + centrals and galaxy counts per halo
            shmr_hydro, bin_cents = get_shmr(inds_top_cent, SubGrNr_fp[inds_top_cent], SubMstar_fp, GrMcrit_fp)
            plot_shmr(shmr_hydro, bin_cents, label='hydro')
            np.save("data/shmr_hydro_"+str(num_gals_hydro)+"_"+type_gal+''+str_snap+'.npy',shmr_hydro)

    plt.legend()
    plt.savefig('figs/SHMR.png')
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
