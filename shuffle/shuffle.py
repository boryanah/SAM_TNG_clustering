#!/usr/bin/env python3
'''
Script for computing and saving shuffled galaxies by mass bin.

Usage:
------
./shuffle.py --help
'''

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

import plotparams # for making plots pretty, but comment out
plotparams.buba() # same
from tools.matching import get_match, match_halos, match_subs
from tools.halostats import get_hist_count, get_xyz_w, get_jack_corr, get_shuff_counts, get_from_sub_prop, get_hmf

np.random.seed(300)

DEFAULTS = {}
DEFAULTS['h'] = 0.6774
DEFAULTS['snapshot'] = 99
DEFAULTS['Lbox'] = 205.
#DEFAULTS['Lbox'] = 75.
if np.abs(DEFAULTS['Lbox'] - 75.) < 1.e-6: sim_name = 'TNG100'
elif np.abs(DEFAULTS['Lbox'] - 205.) < 1.e-6: sim_name = 'TNG300'
DEFAULTS['hydro_dir'] = '/mnt/gosling1/boryanah/'+sim_name+'/'
DEFAULTS['SAM_dir'] = '/mnt/store1/boryanah/SAM_subvolumes_'+sim_name+'/'
DEFAULTS['num_gals'] = 12000 
DEFAULTS['type_gal'] = "mstar" #"mstar"#"sfr"#"mhalo"
DEFAULTS['secondary_property'] = 'shuff'

def main(type_gal, secondary_property, num_gals, want_matching_sam=False, want_matching_hydro=False, record_relative=True, want_norm=False, want_cents=False, h=DEFAULTS['h'], snapshot=DEFAULTS['snapshot'], hydro_dir=DEFAULTS['hydro_dir'], SAM_dir=DEFAULTS['SAM_dir'], Lbox=DEFAULTS['Lbox']):

    # if we are selecting centrals only, need some padding so that we end up with num_gals objects
    if want_cents:
        num_gals *= 2 # padding
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
        order_type = 'mixed' # whatever
        sec_label = 'shuffled'


    ##########################################
    #####               SAM              ##### 
    ##########################################

    # subhalo arrays
    hosthaloid = np.load(SAM_dir+'GalpropHaloIndex'+str_snap+'.npy').astype(int)
    mhalo = np.load(SAM_dir+'GalpropMhalo'+str_snap+'.npy')
    mstar = np.load(SAM_dir+'GalpropMstar'+str_snap+'.npy')
    sfr = np.load(SAM_dir+'GalpropSfr'+str_snap+'.npy')
    sfr /= mstar
    #sfr = np.load(SAM_dir+'GalpropSfrave100myr'+str_snap+'.npy')
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
    halo_tform = np.load(SAM_dir+'GalpropTmerger'+str_snap+'.npy')[sat_type == 0]
    halo_vdiskpeak = np.load(SAM_dir+'HalopropVdisk_peak'+str_snap+'.npy')
    halo_spin = np.load(SAM_dir+'HalopropSpin'+str_snap+'.npy')
    halo_sigma_bulge = np.load(SAM_dir+'GalpropSigmaBulge'+str_snap+'.npy')[sat_type == 0]

    if want_matching_sam or want_matching_hydro:
        # load the matches with hydro
        halo_subfind_id = np.load(SAM_dir+'HalopropSubfindID'+str_snap+'.npy')

    N_halos_sam = len(halo_m_vir)

    # choose only the host halo positions
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
    # might be best
    SubMstar_fp = np.load(hydro_dir+'SubhaloMassInHalfRadType_fp'+str_snap+'.npy')[:,4]*1.e10
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp'+str_snap+'.npy')[:,4]*1.e10
    SubhaloSFR_fp = np.load(hydro_dir+'SubhaloSFR_fp'+str_snap+'.npy')
    SubhaloSFR_fp /= SubMstar_fp
    # TESTING
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassInHalfRadType_fp'+str_snap+'.npy')[:,4]*1.e10
    # og
    #SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp'+str_snap+'.npy')[:,4]*1.e10


    # total number of halos
    N_halos_hydro = len(GrMcrit_fp)

    # environment
    halo_ijk = (GroupPos_fp/gr_size).astype(int)%n_gr
    GroupEnv_fp = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

    #GrRcrit_fp = np.load(hydro_dir+'Group_R_Crit200_fp'+str_snap+'.npy')/1000.
    GrRcrit_fp = np.load(hydro_dir+'Group_R_TopHat200_fp'+str_snap+'.npy')/1000.
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
        print(len(inds_top))
        #inds_top = (np.argsort(mhalo)[::-1])[:num_gals_sam]
        mhalo_low = (np.sort(mhalo)[::-1])[num_gals_sam]
        print("minimum halo mass = ",mhalo_low)

    if want_cents:
        inds_gal = np.arange(len(mstar),dtype=int)
        inds_cent = inds_gal[sat_type == 0]
        bool_top_cent = np.in1d(inds_top, inds_cent)
        print("number of centrals = ",np.sum(bool_top_cent))
        num_gals_sam //= 2
        inds_top = (inds_top[bool_top_cent])[:num_gals_sam]

    if want_norm:
        r_choice = halo_rhalo
        if want_matching_sam:
            r_choice = halo_rhalo[sam_matched]
    else:
        r_choice = None

    if want_matching_sam:
        hmf_sam, bin_hmf = get_hmf(halo_m_vir[sam_matched])
    else:
        hmf_sam, bin_hmf = get_hmf(halo_m_vir)
        
    if want_matching_sam:
        inds_top_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top, hosthaloid, sam_matched, sam_matched)

        print("percentage matches = ",len(inds_top_matched)/num_gals_sam)

        # limit to only matched
        # TESTING
        #inds_top_matched, inds_halo_host_comm1_matched = inds_top_matched[:8000], inds_halo_host_comm1_matched[:8000]

        # total histogram of sats + centrals and galaxy counts per halo
        hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, xyz_position, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], Lbox, group_rad=r_choice, record_relative=record_relative)

        # shuffle halo occupations
        count_nstart_sam = np.vstack((count_sam,nstart_sam)).T
        if halo_prop is not None:
            count_hod, nstart_hod = get_shuff_counts(count_nstart_sam,halo_m_vir[sam_matched], record_relative=record_relative,order_by=halo_prop[sam_matched],order_type=order_type)
        else:
            count_hod, nstart_hod = get_shuff_counts(count_nstart_sam,halo_m_vir[sam_matched],record_relative=record_relative,order_by=halo_prop,order_type=order_type)

    else:
        # total histogram of sats + centrals and galaxy counts per halo
        hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top, hosthaloid[inds_top], xyz_position, halo_m_vir, halo_xyz_position, Lbox, group_rad=r_choice, record_relative=record_relative)

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
    np.save(os.path.expanduser('~/SAM/SAM_TNG_clustering/gm/data_pos/xyz_gals_sam_'+str(num_gals_hydro)+"_"+type_gal+str_snap+'.npy'), xyz_true)

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

    print("Number of galaxies = ", len(w_hod))

    # Correlation function ratios for the SAM

    '''
    # compute the ratio of the corr func of the constructed HOD to the true SAM with error bars and plot it
    Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_cents,w_cents,Lbox)
    plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='--',label='SAM: halo cents/true',alpha=1.,fmt='o',capsize=4)
    '''

    # compute the ratio of the shuffled galaxies to the true SAM with error bars and plot it
    Rat_hodtrue_mean_sam, Rat_hodtrue_err_sam, Corr_mean_hod_sam, Corr_err_hod_sam,  Corr_mean_true_sam, Corr_err_true_sam, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)

    plt.figure(1,figsize=(8,7))

    line = np.linspace(0,40,3)
    plt.plot(line,np.ones(len(line)),'k--')

    plt.errorbar(bin_centers,Rat_hodtrue_mean_sam,yerr=Rat_hodtrue_err_sam,color='dodgerblue',ls='-',label='SAM',alpha=1.,fmt='o',capsize=4)

    np.save("data/rat_mean_sam_"+str(num_gals_sam)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',Rat_hodtrue_mean_sam)
    np.save("data/rat_err_sam_"+str(num_gals_sam)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',Rat_hodtrue_err_sam)

    np.save("data/mean_sam.npy", Corr_mean_true_sam)
    np.save("data/err_sam.npy", Corr_err_true_sam)
    
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

    if want_cents:
        unique_hosts, inds_cent = np.unique(SubGrNr_fp,return_index=True)
        bool_top_cent = np.in1d(inds_top, inds_cent)
        num_gals_hydro //= 2
        inds_top = (inds_top[bool_top_cent])[:num_gals_hydro]
        print("number of centrals = ",np.sum(bool_top_cent))

    if want_norm:
        r_choice = GrRcrit_fp
        if want_matching_hydro:
            r_choice = halo_rhalo[sam_matched]
    else:
        r_choice = None

    if want_matching_hydro:
        hmf_fp, bin_hmf = get_hmf(halo_m_vir[sam_matched])
    else:
        hmf_fp, bin_hmf = get_hmf(GrMcrit_fp)
        
    if want_matching_hydro:
        # limit to only matched
        inds_top_matched, inds_halo_host_comm1_matched, inds_halo_host_matched = match_subs(inds_top, SubGrNr_fp, sam_matched, hydro_matched)
        print("percentage matches = ",len(inds_top_matched)/num_gals_hydro)
        
        # total histogram of sats + centrals and galaxy counts per halo
        hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, SubhaloPos_fp, halo_m_vir[sam_matched], halo_xyz_position[sam_matched], Lbox, group_rad=r_choice, record_relative=record_relative)
        # using hydro centers
        #hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top_matched, inds_halo_host_comm1_matched, SubhaloPos_fp, halo_m_vir[sam_matched], GroupPos_fp[hydro_matched], group_rad=r_choice, record_relative=record_relative)

        count_nstart_fp = np.vstack((count_fp,nstart_fp)).T
        if halo_prop is not None:
            count_hod_fp, nstart_hod_fp = get_shuff_counts(count_nstart_fp,halo_m_vir[sam_matched],record_relative=record_relative,order_by=halo_prop[sam_matched],order_type=order_type)
        else:
            count_hod_fp, nstart_hod_fp = get_shuff_counts(count_nstart_fp,halo_m_vir[sam_matched],record_relative=record_relative,order_by=halo_prop,order_type=order_type)
    else:

        # total histogram of sats + centrals and galaxy counts per halo
        hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top,SubGrNr_fp[inds_top], SubhaloPos_fp, GrMcrit_fp, GroupPos_fp,  Lbox, group_rad=r_choice, record_relative=record_relative)

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
    np.save(os.path.expanduser('~/SAM/SAM_TNG_clustering/gm/data_pos/xyz_gals_hydro'+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy'), xyz_true)
    
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

    print("Number of galaxies = ", len(w_hod))
        
    # Correlation function ratios for the hydro
    
    '''
    # compute the ratio of the corr func of the constructed HOD to the true hydro with error bars and plot it
    Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_cents,w_cents,Lbox)
    plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='orange',ls='--',label='Hydro: halo cents/true',alpha=1.,fmt='o',capsize=4)
    '''

    # compute the ratio of the shuffled galaxies to the true hydro with error bars and plot it
    Rat_hodtrue_mean_hydro, Rat_hodtrue_err_hydro, Corr_mean_hod_hydro, Corr_err_hod_hydro,  Corr_mean_true_hydro, Corr_err_true_hydro, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)

    plt.figure(1,figsize=(8,7))

    # TESTING
    #Rat_hodtrue_mean_hydro = Corr_mean_hod_sam/Corr_mean_hod_hydro
    
    plt.errorbar(bin_centers*1.05,Rat_hodtrue_mean_hydro,yerr=Rat_hodtrue_err_hydro,color='orange',ls='-',label='Hydro',alpha=1.,fmt='o',capsize=4)

    np.save("data/rat_mean_hydro_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',Rat_hodtrue_mean_hydro)
    np.save("data/rat_err_hydro_"+str(num_gals_hydro)+"_"+type_gal+str_cent+"_"+secondary_property+str_snap+'.npy',Rat_hodtrue_err_hydro)

    np.save("data/mean_hydro.npy", Corr_mean_true_hydro)
    np.save("data/err_hydro.npy", Corr_err_true_hydro)
    np.save("data/rel_pos_gals_sam.npy", rel_pos_gals_sam)
    np.save("data/rel_pos_gals_fp.npy", rel_pos_gals_fp)
    np.save("data/bin_centers.npy",bin_centers)
    np.save("data/bin_hmf.npy", bin_hmf)
    np.save("data/hmf_fp.npy", hmf_fp)
    np.save("data/hmf_sam.npy", hmf_sam)
    
    plt.xlabel(r'$r [{\rm Mpc}/h]$')
    plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
    plt.xscale('log')
    plt.legend()
    plt.xlim([0.1,13])
    plt.ylim([0.4,1.5])
    plt.text(0.2,0.5,sec_label)
    plt.savefig("figs/mock_ratio_"+secondary_property+".png")
    

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--type_gal', help='Galaxy type', default=DEFAULTS['type_gal'])
    parser.add_argument('--secondary_property', help='Secondary property', default=DEFAULTS['secondary_property'])
    parser.add_argument('--num_gals', help='Number of galaxies', type=int, default=DEFAULTS['num_gals'])
    parser.add_argument('--want_matching_sam', help='Match SAM?', action='store_true')
    parser.add_argument('--want_matching_hydro', help='Match Hydro?', action='store_true')
    parser.add_argument('--want_cents', help='Work with centrals only?', action='store_true')
    parser.add_argument('--want_norm', help='Normalize the relative positions', action='store_true')
    args = parser.parse_args()
    args = vars(args)
    main(**args)
