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
DEFAULTS['Lbox'] = 205.
if np.abs(DEFAULTS['Lbox'] - 75.) < 1.e-6: sim_name = 'TNG100'
elif np.abs(DEFAULTS['Lbox'] - 205.) < 1.e-6: sim_name = 'TNG300'
DEFAULTS['hydro_dir'] = '/mnt/gosling1/boryanah/'+sim_name+'/'
DEFAULTS['SAM_dir'] = '/mnt/store1/boryanah/SAM_subvolumes_'+sim_name+'/'
DEFAULTS['num_gals'] = 12000
DEFAULTS['type_gal'] = "mstar" #"mstar"#"sfr"#"mhalo"
DEFAULTS['secondary_property'] = 'shuff'

def main(type_gal, secondary_property, num_gals, snapshot, want_matching_sam=False, want_matching_hydro=False, record_relative=True, want_cents=False, want_abundance=False, h=DEFAULTS['h'], hydro_dir=DEFAULTS['hydro_dir'], SAM_dir=DEFAULTS['SAM_dir'], Lbox=DEFAULTS['Lbox']):

    # if we are selecting centrals only, need some padding so that we end up with num_gals objects
    if want_cents or want_abundance:
        # padding
        fac = 3
        num_gals *= fac
        if want_abundance:
            str_cent = ""#"_abund"
            if type_gal == 'mstar':
                f_sat = 0.25
            elif type_gal == 'sfr':
                f_sat = 0.33
        else:
            str_cent = "_cent"
    else:
        str_cent = ""

    if snapshot == 99:
        str_snap = ''
    else:
        str_snap = '_%d'%snapshot
        
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
    hosthaloid = np.load(SAM_dir+'GalpropHaloIndex'+str_snap+'.npy').astype(int)
    mhalo = np.load(SAM_dir+'GalpropMhalo'+str_snap+'.npy')
    mstar = np.load(SAM_dir+'GalpropMstar'+str_snap+'.npy')
    sfr = np.load(SAM_dir+'GalpropSfr'+str_snap+'.npy')
    #sfr = np.load(SAM_dir+'GalpropSfrave100myr'+str_snap+'.npy')
    #sfr /= mstar
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
    halo_tform = np.load(SAM_dir+'Halopropz_Mvir_half'+str_snap+'.npy')
    halo_vdiskpeak = np.load(SAM_dir+'HalopropVdisk_peak'+str_snap+'.npy')
    halo_spin = np.load(SAM_dir+'HalopropSpin'+str_snap+'.npy')
    halo_sigma_bulge = np.load(SAM_dir+'GalpropSigmaBulge'+str_snap+'.npy')[sat_type == 0]

    if want_matching_sam or want_matching_hydro:
        # load the matches with hydro
        #halo_subfind_id = np.load(SAM_dir+'HalopropSubfindID'+str_snap+'.npy')
        # TESTING!!!!!!!!!!!!!! problem at snap = 55
        halo_subfind_id = np.load(SAM_dir+'GalpropSubfindIndex_FP'+str_snap+'.npy')[sat_type==0]

    N_halos_sam = len(halo_m_vir)
    inds_gal = np.arange(len(mstar),dtype=int)

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
    #GrMcrit_fp = np.load(hydro_dir+'Group_M_Crit200_fp'+str_snap+'.npy')*1.e10
    GrMcrit_fp = np.load(hydro_dir+'Group_M_TopHat200_fp'+str_snap+'.npy')*1.e10
    GroupPos_fp = np.load(hydro_dir+'GroupPos_fp'+str_snap+'.npy')/1000.
    SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp'+str_snap+'.npy')/1000.
    # this might be best
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassInRadType_fp'+str_snap+'.npy')[:,4]*1.e10
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassInMaxRadType_fp'+str_snap+'.npy')[:,4]*1.e10
    SubhaloSFR_fp = np.load(hydro_dir+'SubhaloSFR_fp'+str_snap+'.npy')
    #SubhaloSFR_fp /= SubMstar_fp
    # TESTING
    SubMstar_fp = np.load(hydro_dir+'SubhaloMassInHalfRadType_fp'+str_snap+'.npy')[:,4]*1.e10
    # og
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp'+str_snap+'.npy')[:,4]*1.e10



    # total number of halos
    N_halos_hydro = len(GrMcrit_fp)
    unique_hosts, inds_cent = np.unique(SubGrNr_fp,return_index=True)

    # environment
    halo_ijk = (GroupPos_fp/gr_size).astype(int)%n_gr
    GroupEnv_fp = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

    #GrRcrit_fp = np.load(hydro_dir+'Group_R_Crit200_fp'+str_snap+'.npy')
    GrRcrit_fp = np.load(hydro_dir+'Group_R_TopHat200_fp'+str_snap+'.npy')/1000.
    Group_Vmax_fp = np.load(hydro_dir+'Group_Vmax_fp'+str_snap+'.npy')/1000.
    #Group_R_Mean200_fp = np.load(hydro_dir+'Group_R_Mean200_fp'+str_snap+'.npy')/1000
    #Group_V_Crit200_fp = np.load(hydro_dir+'Group_V_Crit200_fp'+str_snap+'.npy')/1000
    Group_V_Crit200_fp = np.load(hydro_dir+'Group_V_TopHat200_fp'+str_snap+'.npy')/1000
    GroupSpin_fp = np.load(hydro_dir+'GroupSpin_fp'+str_snap+'.npy')
    SubhaloVelDisp_fp = np.load(hydro_dir+'SubhaloVelDisp_fp'+str_snap+'.npy')
    SubhaloHalfmassRad_fp = np.load(hydro_dir+'SubhaloHalfmassRad_fp'+str_snap+'.npy')

    if secondary_property == 'env': halo_prop = halo_environment; group_prop = GroupEnv_fp; order_type = 'desc'; sec_label = r'${\rm env. \ (descending)}$'
    elif secondary_property == 'rvir': halo_prop = halo_rhalo; group_prop = GrRcrit_fp; order_type = 'desc'; sec_label = r'${\rm vir. \ rad. \ (descending)}$'
    elif secondary_property == 'mpeak': halo_prop = halo_mpeak; group_prop = np.zeros(len(GrRcrit_fp)); order_type = 'desc'; sec_label = r'${\rm peak mass \ (descending)}$'
    elif secondary_property == 'vdiskpeak': halo_prop = halo_vdiskpeak; group_prop = np.zeros(len(GrRcrit_fp)); order_type = 'desc'; sec_label = r'${\rm peak disk vel. \ (descending)}$'
    elif secondary_property == 'tform': halo_prop = halo_tform; group_prop = np.zeros(len(GrRcrit_fp)); order_type = 'desc'; sec_label = r'${\rm formation time \ (descending)}$'
    elif secondary_property == 'conc': halo_prop = halo_c_nfw; group_prop = (Group_Vmax_fp/Group_V_Crit200_fp); order_type = 'mixed'; sec_label = r'${\rm conc. \ (mixed)}$'
    elif secondary_property == 'vdisp': halo_prop = halo_sigma_bulge; group_prop = get_from_sub_prop(SubhaloVelDisp_fp, SubGrNr_fp, N_halos_hydro); order_type = 'mixed'; sec_label = r'${\rm vel. \ disp. \ (mixed)}$'
    elif secondary_property == 'spin': halo_prop = halo_spin; group_prop = GroupSpin_fp; order_type = 'desc'; sec_label = r'${\rm spin \ (descending)}$'
    elif secondary_property == 's2r': halo_prop = halo_sigma_bulge**2*halo_rhalo; group_prop = get_from_sub_prop(SubhaloVelDisp_fp**2*SubhaloHalfmassRad_fp, SubGrNr_fp, N_halos_hydro); order_type = 'asc'; sec_label = r'${\rm spin \ (descending)}$'

    # --------------------------------------------------------
    #                   MATCH SAM HYDRO
    # --------------------------------------------------------

    if want_matching_sam or want_matching_hydro:
        # obtain matching indices for halos
        hydro_matched, sam_matched = get_match(halo_subfind_id, SubGrNr_fp, N_halos_sam)
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
        mhalo_low = (np.sort(mhalo)[::-1])[num_gals_sam]

    if want_cents:
        # selecting centrals
        inds_gal = np.arange(len(mstar),dtype=int)
        inds_cent = inds_gal[sat_type == 0]
        bool_top_cent = np.in1d(inds_top, inds_cent)
        print("number of centrals = ",np.sum(bool_top_cent))
        num_gals_sam //= fac
        inds_top = (inds_top[bool_top_cent])[:num_gals_sam]
        
    if want_abundance:
        # selecting centrals and satellites
        inds_gal = np.arange(len(mstar), dtype=int)
        inds_cent = inds_gal[sat_type == 0]
        inds_sats = inds_gal[sat_type != 0]
        bool_top_cent = np.in1d(inds_top, inds_cent)
        bool_top_sats = np.in1d(inds_top, inds_sats)
        print("SAM:")
        print("number of centrals = ",np.sum(bool_top_cent))
        print("number of satellites = ",np.sum(bool_top_sats))
        num_gals_sam //= fac
        num_sats_sam = int(np.round(f_sat*num_gals_sam))
        num_cent_sam = num_gals_sam - num_sats_sam
        inds_top_cent = (inds_top[bool_top_cent])[:num_cent_sam]
        inds_top_sats = (inds_top[bool_top_sats])[:num_sats_sam]
        print("number of centrals = ", len(inds_top_cent))
        print("number of satellites = ", len(inds_top_sats))
        inds_top = np.hstack((inds_top_cent, inds_top_sats))


    if want_matching_sam:
        inds_top_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top, hosthaloid, sam_matched, sam_matched)

        # total histogram of sats + centrals and galaxy counts per halo
        hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, xyz_position, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], Lbox, record_relative=record_relative)

        # centrals
        inds_top_matched, comm1, comm2 = np.intersect1d(inds_gal[sat_type == 0], inds_top_matched, return_indices=True)
        inds_halo_host_comm1_matched = inds_halo_host_comm1_matched[comm2]
        hist_cents_sam, bin_cents, count_cents_sam, nstart_cents_sam, rel_pos_gals_cents_sam = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, xyz_position, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], Lbox, record_relative=record_relative)

        if halo_prop is not None:
            hist_sam_top, hist_sam_bot, bin_cents = get_hod_prop(halo_m_vir[sam_matched],count_sam,halo_prop[sam_matched])
            plot_hod_prop(hist_sam_top, hist_sam_bot, bin_cents, label='SAM')
            np.save("data/hist_sam_top_"+str(int(num_gals_hydro/fac))+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_sam_top)
            np.save("data/hist_sam_bot_"+str(int(num_gals_hydro/fac))+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_sam_bot)

            # centrals
            hist_cents_sam_top, hist_cents_sam_bot, bin_cents = get_hod_prop(halo_m_vir[sam_matched],count_cents_sam,halo_prop[sam_matched])
            np.save("data/hist_cents_sam_top_"+str(int(num_gals_hydro/fac))+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_cents_sam_top)
            np.save("data/hist_cents_sam_bot_"+str(int(num_gals_hydro/fac))+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_cents_sam_bot)
        else:
            hist_sam, bin_cents = get_hod(halo_m_vir[sam_matched],count_sam)
            plot_hod(hist_sam, bin_cents,label='SAM')
            np.save("data/hist_sam_"+str(int(num_gals_hydro/fac))+"_"+type_gal+str_cent+str_snap+'.npy',hist_sam)

            # centrals
            hist_cents_sam, bin_cents = get_hod(halo_m_vir[sam_matched],count_cents_sam)
            np.save("data/hist_cents_sam_"+str(int(num_gals_hydro/fac))+"_"+type_gal+str_cent+str_snap+'.npy',hist_cents_sam)

    else:
        # total histogram of sats + centrals and galaxy counts per halo
        hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top, hosthaloid[inds_top], xyz_position, halo_m_vir, halo_xyz_position, Lbox, record_relative=record_relative)#, other_group_mass=GrMcrit_fp) # TESTING tuks

        # centrals
        inds_top = np.intersect1d(inds_gal[sat_type == 0], inds_top)
        hist_cents_sam, bin_cents, count_cents_sam, nstart_cents_sam, rel_pos_gals_cents_sam = get_hist_count(inds_top, hosthaloid[inds_top], xyz_position, halo_m_vir, halo_xyz_position, Lbox, record_relative=record_relative)#, other_group_mass=GrMcrit_fp) # TESTING tuks
        
        if halo_prop is not None:
            hist_sam_top, hist_sam_bot, bin_cents = get_hod_prop(halo_m_vir,count_sam,halo_prop)
            plot_hod_prop(hist_sam_top, hist_sam_bot, bin_cents, label='SAM')
            np.save("data/hist_sam_top_"+str(num_gals_sam)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_sam_top)
            np.save("data/hist_sam_bot_"+str(num_gals_sam)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_sam_bot)

            # centrals
            hist_cents_sam_top, hist_cents_sam_bot, bin_cents = get_hod_prop(halo_m_vir,count_cents_sam,halo_prop)
            np.save("data/hist_cents_sam_top_"+str(num_gals_sam)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_cents_sam_top)
            np.save("data/hist_cents_sam_bot_"+str(num_gals_sam)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_cents_sam_bot)
        else:
            # TESTING tuks
            hist_sam, bin_cents = get_hod(halo_m_vir, count_sam)#, other_group_mass=GrMcrit_fp) # TESTING
            plot_hod(hist_sam, bin_cents,label='SAM')
            np.save("data/hist_sam_"+str(num_gals_sam)+"_"+type_gal+str_cent+str_snap+'.npy',hist_sam)

            # centrals
            # TESTING tuks
            hist_cents_sam, bin_cents = get_hod(halo_m_vir,count_cents_sam)#, other_group_mass=GrMcrit_fp) # TESTING
            np.save("data/hist_cents_sam_"+str(num_gals_sam)+"_"+type_gal+str_cent+str_snap+'.npy',hist_cents_sam)

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
        bool_arr = np.in1d(SubGrNr_fp, hydro_matched[inds_halo_top_matched])
        bool_arr = (SubMstar_fp > mstar_low) & bool_arr
        inds_top = np.arange(len(SubMstar_fp))[bool_arr]
        print(len(inds_top))

    if want_cents:
        # selecting centrals
        unique_hosts, inds_cent = np.unique(SubGrNr_fp,return_index=True)
        bool_top_cent = np.in1d(inds_top, inds_cent)
        num_gals_hydro //= fac
        inds_top = (inds_top[bool_top_cent])[:num_gals_hydro]
        print("number of centrals = ",np.sum(bool_top_cent))

    if want_abundance:
        # selecting centrals and satellites
        unique_hosts, inds_cent = np.unique(SubGrNr_fp, return_index=True)
        inds_gal = np.arange(len(SubGrNr_fp), dtype=int)
        inds_sats = inds_gal[np.in1d(inds_gal, inds_cent, invert=True)]
        bool_top_cent = np.in1d(inds_top, inds_cent)
        bool_top_sats = np.in1d(inds_top, inds_sats)
        print("TNG:")
        print("number of centrals = ",np.sum(bool_top_cent))
        print("number of satellites = ",np.sum(bool_top_sats))
        num_gals_hydro //= fac
        num_sats_hydro = int(np.round(f_sat*num_gals_hydro))
        num_cent_hydro = num_gals_hydro - num_sats_hydro
        inds_top_cent = (inds_top[bool_top_cent])[:num_cent_hydro]
        inds_top_sats = (inds_top[bool_top_sats])[:num_sats_hydro]
        print("number of centrals = ", len(inds_top_cent))
        print("number of satellites = ", len(inds_top_sats))
        inds_top = np.hstack((inds_top_cent, inds_top_sats))
        
    if want_matching_hydro:
        inds_top_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top, SubGrNr_fp, sam_matched, hydro_matched)

        # total histogram of sats + centrals and galaxy counts per halo
        hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, SubhaloPos_fp, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], Lbox, record_relative=record_relative)

        # centrals
        inds_top_matched, comm1, comm2 = np.intersect1d(inds_cent, inds_top_matched, return_indices=True)
        inds_halo_host_comm1_matched = inds_halo_host_comm1_matched[comm2]
        hist_cents_hydro, bin_cents, count_cents_fp, nstart_cents_fp, rel_pos_gals_cents_fp = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, SubhaloPos_fp, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], Lbox, record_relative=record_relative)

        if halo_prop is not None:
            hist_hydro_top, hist_hydro_bot, bin_cents = get_hod_prop(halo_m_vir[sam_matched], count_fp, halo_prop[sam_matched])
            plot_hod_prop(hist_hydro_top, hist_hydro_bot, bin_cents, label='hydro')
            np.save("data/hist_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_hydro_top)
            np.save("data/hist_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_hydro_bot)

            # centrals
            hist_cents_hydro_top, hist_cents_hydro_bot, bin_cents = get_hod_prop(halo_m_vir[sam_matched],count_cents_fp,halo_prop[sam_matched])
            np.save("data/hist_cents_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_cents_hydro_top)
            np.save("data/hist_cents_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_cents_hydro_bot)
        else:
            hist_hydro, bin_cents = get_hod(halo_m_vir[sam_matched],count_fp)
            plot_hod(hist_hydro, bin_cents, label='hydro')
            np.save("data/hist_hydro_"+str(num_gals_hydro)+"_"+type_gal+str_cent+str_snap+'.npy',hist_hydro)

            # centrals
            hist_cents_hydro, bin_cents = get_hod(halo_m_vir[sam_matched],count_cents_fp)
            np.save("data/hist_cents_hydro_"+str(num_gals_hydro)+"_"+type_gal+str_cent+str_snap+'.npy',hist_cents_hydro)

    else:
        # total histogram of sats + centrals and galaxy counts per halo
        hist_hydro, bin_cents, count_fp, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top, SubGrNr_fp[inds_top], SubhaloPos_fp, GrMcrit_fp, GroupPos_fp, Lbox, record_relative=record_relative)

        # centrals
        inds_top = np.intersect1d(inds_cent, inds_top)
        hist_cents_hydro, bin_cents, count_cents_fp, nstart_cents_fp, rel_pos_gals_cents_fp = get_hist_count(inds_top, SubGrNr_fp[inds_top], SubhaloPos_fp, GrMcrit_fp, GroupPos_fp, Lbox, record_relative=record_relative)

        if halo_prop is not None:
            hist_hydro_top, hist_hydro_bot, bin_cents = get_hod_prop(GrMcrit_fp,count_fp,group_prop)
            plot_hod_prop(hist_hydro_top, hist_hydro_bot, bin_cents, label='hydro')
            np.save("data/hist_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_hydro_top)
            np.save("data/hist_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_hydro_bot)

            # centrals
            hist_cents_hydro_top, hist_cents_hydro_bot, bin_cents = get_hod_prop(GrMcrit_fp,count_cents_fp,group_prop)
            np.save("data/hist_cents_hydro_top_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_cents_hydro_top)
            np.save("data/hist_cents_hydro_bot_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',hist_cents_hydro_bot)
        else:
            hist_hydro, bin_cents = get_hod(GrMcrit_fp,count_fp)
            plot_hod(hist_hydro, bin_cents, label='hydro')
            np.save("data/hist_hydro_"+str(num_gals_hydro)+"_"+type_gal+str_cent+str_snap+'.npy',hist_hydro)

            # centrals
            hist_cents_hydro, bin_cents = get_hod(GrMcrit_fp,count_cents_fp)
            np.save("data/hist_cents_hydro_"+str(num_gals_hydro)+"_"+type_gal+str_cent+str_snap+'.npy',hist_cents_hydro)

    plt.legend()
    plt.ylim([1.e-6, 40])
    plt.savefig('figs/HOD_'+secondary_property+'_'+type_gal+str_cent+'.png')
    #plt.show()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--type_gal', help='Galaxy type', default=DEFAULTS['type_gal'])
    parser.add_argument('--secondary_property', help='Secondary property', default=DEFAULTS['secondary_property'])
    parser.add_argument('--num_gals', help='Number of galaxies', type=int, default=DEFAULTS['num_gals'])
    parser.add_argument('--snapshot', help='Simulation snapshot', type=int, default=DEFAULTS['snapshot'])
    parser.add_argument('--want_matching_sam', help='Match SAM?', action='store_true')
    parser.add_argument('--want_matching_hydro', help='Match Hydro?', action='store_true')
    parser.add_argument('--want_cents', help='Work with centrals only?', action='store_true')
    parser.add_argument('--want_abundance', help='Work with centrals and satellites', action='store_true')
    args = parser.parse_args()
    args = vars(args)
    main(**args)
