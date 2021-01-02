import matplotlib.pyplot as plt
import numpy as np
import plotparams # for making plots pretty, but comment out
#plotparams.buba() # same
import Corrfunc # comment out since you're not using it
import sys

# REMOVE ME
import matplotlib.style as style
style.use('seaborn-deep')
style.use('seaborn-poster')
plt.rcParams['font.family'] = "serif"


# This is a function that takes the position of the unshuffled galaxies, their weights (which is to say all 1s if you are working with the actual galaxy positions or the galaxy counts in halo if putting all galaxies at the center of a halo), the positions of the shuffled galaxies, their weights and the box size. It then returns the bin centers, mean of the ratio between the two (shuffled-to-unshuffled) and errorbars of the ratio (it also returns the correlation functions and error bars for both, but I generally don't use those)
def get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox):
    
    # bins for the correlation function
    N_bin = 21
    bins = np.logspace(-1.,1.,N_bin)
    bin_centers = (bins[:-1] + bins[1:])/2.

    # dimensions for jackknifing
    N_dim = 3

    # empty arrays to record data
    Rat_hodtrue = np.zeros((N_bin-1,N_dim**3))
    Corr_hod = np.zeros((N_bin-1,N_dim**3))
    Corr_true = np.zeros((N_bin-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                xyz_hod_jack = xyz_hod.copy()
                xyz_true_jack = xyz_true.copy()
                w_hod_jack = w_hod.copy()
                w_true_jack = w_true.copy()
            
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_hod/size).astype(int)),axis=1).astype(bool)
                xyz_hod_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_hod_jack = xyz_hod_jack[np.sum(xyz_hod_jack,axis=1)!=0.]
                w_hod_jack[bool_arr] = -1
                w_hod_jack = w_hod_jack[np.abs(w_hod_jack+1) > 1.e-6]

                bool_arr = np.prod((xyz == (xyz_true/size).astype(int)),axis=1).astype(bool)
                xyz_true_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_true_jack = xyz_true_jack[np.sum(xyz_true_jack,axis=1)!=0.]
                w_true_jack[bool_arr] = -1
                w_true_jack = w_true_jack[np.abs(w_true_jack+1) > 1.e-6]

                # !!! Important !!! would just have to say here:
                #res_hod = get_corrfunc(xyz_hod_jack,bins,Lbox)
                #res_true = get_corrfunc(xyz_true_jack,bins,Lbox)
                # However, if the weights are not ones (i.e. we're placing galaxies at the centers of halos, then you would need to definte your own array with repetitions for halos with multiple objects; i.e. if halo #2 located at position B has 5 galaxies, you would have xyz_hod_jack = [B,B,B,B,B])

                #res_hod = Corrfunc.theory.xi(X=xyz_hod_jack[:,0],Y=xyz_hod_jack[:,1],Z=xyz_hod_jack[:,2],boxsize=Lbox,weights=w_hod_jack,weight_type='pair_product',nthreads=16,binfile=bins)['xi']
                # for some reason this solves a weird computational bug sometimes...
                if np.abs(np.sum(w_hod_jack)-len(w_hod_jack)) < 1.e-6:
                    res_hod = Corrfunc.theory.xi(X=xyz_hod_jack[:,0],Y=xyz_hod_jack[:,1],Z=xyz_hod_jack[:,2],boxsize=Lbox,nthreads=16,binfile=bins)['xi']
                else:
                    res_hod = Corrfunc.theory.xi(X=xyz_hod_jack[:,0],Y=xyz_hod_jack[:,1],Z=xyz_hod_jack[:,2],boxsize=Lbox,weights=w_hod_jack,weight_type='pair_product',nthreads=16,binfile=bins)['xi']
                res_true = Corrfunc.theory.xi(X=xyz_true_jack[:,0],Y=xyz_true_jack[:,1],Z=xyz_true_jack[:,2],boxsize=Lbox,weights=w_true_jack,weight_type='pair_product',nthreads=16,binfile=bins)['xi']

                rat_hodtrue = res_hod/res_true
                Rat_hodtrue[:,i_x+N_dim*i_y+N_dim**2*i_z] = rat_hodtrue
                Corr_hod[:,i_x+N_dim*i_y+N_dim**2*i_z] = res_hod
                Corr_true[:,i_x+N_dim*i_y+N_dim**2*i_z] = res_true

    # compute mean and error
    Rat_hodtrue_mean = np.mean(Rat_hodtrue,axis=1)
    Rat_hodtrue_err = np.sqrt(N_dim**3-1)*np.std(Rat_hodtrue,axis=1)
    Corr_mean_hod = np.mean(Corr_hod,axis=1)
    Corr_err_hod = np.sqrt(N_dim**3-1)*np.std(Corr_hod,axis=1)
    Corr_mean_true = np.mean(Corr_true,axis=1)
    Corr_err_true = np.sqrt(N_dim**3-1)*np.std(Corr_true,axis=1)

    return Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers

# CAN IGNORE: this function draws number of total galaxies, centrals and satellites for each halo from 0 to N_halos-1, given mass group_mass; it takes the HOD of the total, centrals and satellites (hist, hist_cents, hist_sats) and returns the number of galaxies living in each halo count_halo_hod
def get_hod_counts(hist,hist_cents,hist_sats,N_halos,group_mass):
    # define mass bins
    log_min = 10.
    log_max = 15.
    N_bins = 11
    bin_edges = np.linspace(log_min,log_max,N_bins)
    bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

    count_halo_hod = np.zeros(N_halos,dtype=int)
    count_halo_cents_hod = np.zeros(N_halos,dtype=int)
    count_halo_sats_hod = np.zeros(N_halos,dtype=int)
    for i in range(N_bins-1):
        M_lo = 10.**bin_edges[i]
        M_hi = 10.**bin_edges[i+1]

        mass_sel = (M_lo < group_mass) & (M_hi > group_mass)
        N = int(np.sum(mass_sel))
        # if no halos in mass bin, skip
        if N == 0: continue

        # how many galaxies do we roughly expect in this mass bin
        N_g = N*hist[i]

        prob_cen = hist_cents[i]
        # if the probability is not a real number, i.e. nan or infinity, skip
        if not np.isfinite(prob_cen): continue
        # draw number of centrals from a binomial distribution
        cens = np.random.binomial(1,prob_cen,N)
        N_c = int(np.sum(cens > 0.))
        # draw number of satellites from a poisson distribution
        sats = np.random.poisson(hist_sats[i],N)
        N_s = np.sum(sats)
        N_t = N_c+N_s
        
        print("for this mass bin: total number of gals assigned in halo vs. expected number (out of all N halos in the mass bin) = ",N_t,N_g,N)
        print("_____________________________________")
        tot = sats+cens
        count_halo_hod[mass_sel] = tot

    print("HOD total gals assigned = ",np.sum(count_halo_hod))

    return count_halo_hod

# this function shuffles the number of total galaxies, for each mass bin and returns the number of galaxies living in each halo count_halo_hod
def get_shuff_counts(count_halo,group_mass,record_relative=False,order_by=None,order_type='desc'):
    # define mass bins
    log_min = 10.
    log_max = 15.
    N_bins = 11
    bin_edges = np.linspace(log_min,log_max,N_bins)
    bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

    if record_relative:
        assert count_halo.shape[1] == 2, "you sure?"

    N_halos = len(count_halo)
    
    if record_relative:
        nstart_halo_hod = np.zeros(N_halos,dtype=int)
        count_halo_hod = count_halo[:,0].copy()
    else:
        count_halo_hod = count_halo.copy()

    for i in range(N_bins-1):
        M_lo = 10.**bin_edges[i]
        M_hi = 10.**bin_edges[i+1]


        mass_sel = (M_lo < group_mass) & (M_hi > group_mass)
        inds_this = np.arange(np.sum(mass_sel))
        count_this = count_halo[mass_sel]

        n_bin = np.sum(mass_sel)
        print("halos in bin = ",n_bin)
        try:
            # order by count and by property and give highest
            # number of galaxies to object with highest property
            prop_this = order_by[mass_sel]
            gal_this = count_this[:,0]
            if order_type == 'desc':
                inds_this = (np.argsort(gal_this)[::-1])[np.argsort(np.argsort(prop_this)[::-1])]
            elif order_type == 'asc':
                inds_this = (np.argsort(gal_this)[::-1])[np.argsort(np.argsort(prop_this))]
            elif order_type == 'mixed':
                #print("Not implemented"); quit()
                only_centrals_this = count_this <= 1
                if np.sum(count_this) == 0:
                    # do ascending if all halos in the mass bin have more than one galaxy
                    inds_this = (np.argsort(gal_this)[::-1])[np.argsort(np.argsort(prop_this))]
                else:
                    # split into halos above and below
                    mass_this = group_mass[mass_sel]
                    cent_max = np.max(mass_this[only_centrals_this])
                    smaller = mass_this <= cent_max
                    larger = mass_this > cent_max
                    inds_this[smaller] = (np.argsort(gal_this[smaller])[::-1])[np.argsort(np.argsort(prop_this[smaller])[::-1])]
                    inds_this[larger] = (np.argsort(gal_this[larger])[::-1])[np.argsort(np.argsort(prop_this[larger]))]
        except:
            np.random.shuffle(inds_this)

        count_this = count_this[inds_this]

        if record_relative:
            nstart_this = count_this[:,1]
            count_this = count_this[:,0]
            nstart_halo_hod[mass_sel] = nstart_this
        count_halo_hod[mass_sel] = count_this
        
    print("HOD total gals = ",np.sum(count_halo_hod))
    
    if record_relative:
        return count_halo_hod, nstart_halo_hod
    return count_halo_hod

# this function returns the HOD, bin centers of the HOD and most importantly galaxy counts per halo for subhalos with indices inds_top
def get_hist_count(inds_top,sub_parent_id, sub_pos, group_mass, group_pos, N_halos,record_relative=False):
    
    # Find the halo ID's of their halo parents in the original group_mass array
    hosts_top = sub_parent_id[inds_top]

    pos_top = sub_pos[inds_top]
    # Find the masses of their halo parents from the original group_mass array
    masses_top = group_mass[hosts_top]
    # Knowing the ID's of the relevant halos (i.e. those who are hosting a galaxy),
    # tell me which ID's are unique, what indices in the hosts_top array these
    # unique ID's correspond to and how many times they each repeat
    hosts, inds, gal_counts = np.unique(hosts_top,return_index=True,return_inverse=False, return_counts=True)

    # get unique masses of hosts to compute the HOD (hist)
    host_masses = masses_top[inds]
    hist, bin_cents = get_hod(host_masses,gal_counts,group_mass)

    # get the galaxy counts per halo
    count_halo = np.zeros(N_halos,dtype=int)
    count_halo[hosts] += gal_counts

    if record_relative:
        nstart_halo = np.zeros(N_halos,dtype=int)-1
        rel_pos_gals_halo = np.zeros((len(inds_top),3))
        cumulative = 0
        for i in range(len(hosts)):
            group_cen = group_pos[hosts[i]]
            n_gal = gal_counts[i]

            gal_sel = hosts[i] == hosts_top
            assert np.sum(gal_sel) == n_gal, "you must have messed up"
            rel_pos_gals_halo[cumulative:cumulative+n_gal] = pos_top[gal_sel]-group_cen
            nstart_halo[hosts[i]] = cumulative
            
            cumulative += n_gal
        print("total number of galaxies = ",cumulative)
        #print(rel_pos_gals_halo[:100]);quit()
        return hist, bin_cents, count_halo, nstart_halo, rel_pos_gals_halo

    
    return hist, bin_cents, count_halo

# returns the xyz position and the weights; also works if feeding relative positions
def get_xyz_w(count_sel,nstart_sel,xyz_all,rel_pos_gals,dtype,Lbox):
    xyz_cents_sel = xyz_all[count_sel > 0]
    nstart_sel = nstart_sel[count_sel > 0]
    count_sel = count_sel[count_sel > 0]

    xyz_gal = np.zeros((num_gals,3),dtype=dtype)
    w_gal = np.ones(num_gals,dtype=dtype)
    for i in range(len(count_sel)):
        nst = nstart_sel[i]
        ncs = count_sel[i]
        xyz_center = xyz_cents_sel[i]
        xyz_gal[nst:nst+ncs] = xyz_center+rel_pos_gals[nst:nst+ncs]

    
    xyz_gal[xyz_gal >= Lbox] -= Lbox
    xyz_gal[xyz_gal < 0.] += Lbox 
    return xyz_gal, w_gal

# this function takes the number of galaxies per halo (counts), masses of each halo that has at least one galaxy (masses) and masses of all host halos (which is to say big halos) regardless of whether or not they host one of our 6000 galaxies (all_masses). It then returns the HOD as well as the bin centers
def get_hod(masses,counts,all_masses):
    # number of bin edges
    n_bins = 31
    # bin edges (btw Delta = np.log10(bins[1])-np.log10(bins[0]) in log space)
    bins = np.logspace(10.,15.,n_bins)
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])
    # This histogram tells you how many halos there are in each bin inerval (n_bins-1)
    hist_norm, edges = np.histogram(all_masses,bins=bins)
    hist_weighted, edges = np.histogram(masses,bins=bins,weights=counts)
    hist_hod = hist_weighted/hist_norm

    return hist_hod, bin_cents

# this function takes the number of galaxies per halo (counts), masses of each halo that has at least one galaxy (masses) and masses of all host halos (which is to say big halos) regardless of whether or not they host one of our 6000 galaxies (all_masses), as well as the label you want to use for plotting the HOD. It returns a plot of the HOD function
def plot_hod(masses,counts,all_masses,label):
    hist_hod, bin_cents = get_hod(masses,counts,all_masses)
    # Plot the derived HOD shape
    plt.plot(bin_cents,hist_hod,label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\langle N_{\rm gal} \rangle$')
    plt.xlabel(r'$M_{\rm halo}$')
    plt.text(1.e13,0.05,r'$n_{\rm gal} = %.4f$'%n_gal)
    plt.ylim([1.e-2,100.])
    plt.xlim([1.e10,1.e15])

# CAN IGNORE: this function takes the masses of all host halos (regardless of whether or not they host one of the 6000 galaxies) and it plots number of halos as a function of halo mass -- don't worry about doing this
def plot_hmf(all_masses,label):
    # number of bin edges
    n_bins = 31
    bins = np.logspace(8.,15.,n_bins)
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])

    hist, edges = np.histogram(all_masses,bins=bins)

    plt.plot(bin_cents,hist,label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$N_{\rm halo}$')
    plt.xlabel(r'$M_{\rm halo} [M_\odot/h]$')

def get_from_sub_prop(sub_prop,sub_grnr,n_halo):
    gr_prop = np.zeros(n_halo)
    # gives the unique indices of first occurrence ordered
    unique, inds = np.unique(sub_grnr,return_index=True)
    gr_prop[unique] = sub_prop[inds]
    return gr_prop
    
# Hubble constant, snapshot number (99 is last snapshot, i.e. z = 0), number of galaxies
h = 0.6774
k = 99
num_gals = 6000
Lbox = 75.
n_gal = num_gals/Lbox**3.
record_relative = 1
try:
    secondary_property = sys.argv[1]
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
# subhalo arrays
SAM_dir = '/mnt/gosling1/boryanah/SAM/TNG100-1-SAM-Recalibrated/'
hosthaloid = np.load(SAM_dir+'hosthaloid_%d.npy'%k).astype(int)
mhalo = np.load(SAM_dir+'mhalo_%d.npy'%k)
mstar = np.load(SAM_dir+'mstar_%d.npy'%k)
sat_type = np.load(SAM_dir+'sat_type_%d.npy'%k)
x_position = np.load(SAM_dir+'x_position_%d.npy'%k)
y_position = np.load(SAM_dir+'y_position_%d.npy'%k)
z_position = np.load(SAM_dir+'z_position_%d.npy'%k)

# combine the halo positions into one array
xyz_position = np.vstack((x_position,y_position,z_position)).T
xyz_position[xyz_position > Lbox] -= Lbox
xyz_position[xyz_position < 0.] += Lbox 

# halo arrays
halo_m_vir = np.load(SAM_dir+'halo_m_vir_%d.npy'%k)
halo_c_nfw = np.load(SAM_dir+'halo_c_nfw_%d.npy'%k)
halo_rhalo = np.load(SAM_dir+'halo_rhalo_%d.npy'%k)
halo_spin = np.load(SAM_dir+'halo_spin_%d.npy'%k)
halo_sigma_bulge = np.load(SAM_dir+'halo_sigma_bulge_%d.npy'%k)
N_halos_sam = len(halo_m_vir)

# positions of the halos
halo_xyz_position = xyz_position[sat_type==0]

# environment parameter
density = np.load('../data/smoothed_mass_in_area.npy')
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
GrRcrit_fp = np.load(hydro_dir+'Group_R_Crit200_fp.npy')
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp.npy')/1000.
SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp.npy')/1000.
SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp.npy')[:,4]*1.e10


# total number of halos
N_halos_hydro = len(GrMcrit_fp)

# environment
halo_ijk = (GroupPos_fp/gr_size).astype(int)%n_gr
GroupEnv_fp = density[halo_ijk[:,0],halo_ijk[:,1],halo_ijk[:,2]]

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
#                         SAM
# --------------------------------------------------------

# order the subhalos in terms of their stellar masses
inds_top = (np.argsort(mstar)[::-1])[:num_gals]    

# total histogram of sats + centrals and galaxy counts per halo
hist_sam, bin_cents, count_sam, nstart_sam, rel_pos_gals_sam = get_hist_count(inds_top,hosthaloid, xyz_position, halo_m_vir, halo_xyz_position, N_halos_sam,record_relative=record_relative)

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
w_true = np.ones(num_gals,dtype=xyz_true.dtype)

# here we select out of our newly constructed array with counts per halo only those halos which have galaxies and put the galaxies in the center (weights are not all ones)
xyz_cents = halo_xyz_position[count_sam>0]
w_cents = count_sam[count_sam>0].astype(xyz_cents.dtype)

# version 1 -- put in center
# this is the shuffled HOD where again we put the galaxies at the center of the halos
#xyz_hod = halo_xyz_position[count_hod > 0]
#w_hod = count_hod[count_hod > 0].astype(xyz_hod.dtype)
# version 2 -- put relative positions
xyz_hod, w_hod = get_xyz_w(count_hod,nstart_hod,halo_xyz_position,rel_pos_gals_sam,xyz_cents.dtype,Lbox)


# REMOVE ME
fig, ax = plt.subplots()

# og
plt.figure(1,figsize=(8,7))

line = np.linspace(0,40,3)
plt.plot(line,np.ones(len(line)),'k--')

# Correlation function ratios for the SAM

'''
# compute the ratio of the corr func of the constructed HOD to the true SAM with error bars and plot it
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_cents,w_cents,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='--',label='SAM: halo cents/true',alpha=1.,fmt='o',capsize=4)
'''

# compute the ratio of the shuffled galaxies to the true SAM with error bars and plot it
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='-',label='SAM',alpha=1.,fmt='o',capsize=4)


np.save("data_rat/rat_mean_sam_"+secondary_property+".npy",Rat_hodtrue_mean)
np.save("data_rat/rat_err_sam_"+secondary_property+".npy",Rat_hodtrue_err)

# --------------------------------------------------------
#                         HYDRO
# --------------------------------------------------------


# Find the indices of the top galaxies
inds_top = np.argsort(SubMstar_fp)[::-1][:num_gals]

# total histogram of sats + centrals and galaxy counts per halo
hist_hydro, bin_cents, count_fp, nstart_fp, rel_pos_gals_fp = get_hist_count(inds_top,SubGrNr_fp, SubhaloPos_fp, GrMcrit_fp, GroupPos_fp, N_halos_hydro, record_relative=record_relative)

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
xyz_cents = GroupPos_fp[count_fp > 0]
w_cents = count_fp[count_fp > 0].astype(xyz_cents.dtype)

# version 1 -- put in centers of halos
# this is the shuffled HOD where again we put the galaxies at the center of the halos
#xyz_hod = GroupPos_fp[count_hod_fp > 0]
#w_hod = count_hod_fp[count_hod_fp > 0].astype(xyz_hod.dtype)
# version 2 -- put relative positions
xyz_hod, w_hod = get_xyz_w(count_hod_fp,nstart_hod_fp,GroupPos_fp,rel_pos_gals_fp,xyz_cents.dtype,Lbox)

plt.figure(1,figsize=(8,7))

# Correlation function ratios for the hydro

'''
# compute the ratio of the corr func of the constructed HOD to the true hydro with error bars and plot it
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_cents,w_cents,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='orange',ls='--',label='Hydro: halo cents/true',alpha=1.,fmt='o',capsize=4)
'''

# compute the ratio of the shuffled galaxies to the true hydro with error bars and plot it
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)
plt.errorbar(bin_centers*1.05,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='orange',ls='-',label='Hydro',alpha=1.,fmt='o',capsize=4)

np.save("data_rat/rat_mean_hydro_"+secondary_property+".npy",Rat_hodtrue_mean)
np.save("data_rat/rat_err_hydro_"+secondary_property+".npy",Rat_hodtrue_err)

np.save("data_rat/bin_centers.npy",bin_centers)

# REMOVE ME
plt.legend()
plt.xscale("log")
ax.tick_params(axis="both", which="both", length=0, width=0)
ax.grid(False)
ax.set_xlim(10**-1, 10**1)
plt.xlabel(r"$r \ [\frac{Mpc}{h}]$")
plt.ylabel(r"$\xi_{\rm shuffled}/\xi_{\rm TNG300}$")
plt.savefig("figs/mock_ratio_"+secondary_property+".png")
plt.show()
quit()

# og
plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.xlim([0.1,13])
plt.ylim([0.4,1.5])
plt.text(0.2,0.5,sec_label)
plt.savefig("figs/mock_ratio_"+secondary_property+".png")

quit()
plt.figure(2,figsize=(8,7))

# Plot the HOD for the SAM
#plot_hod(halo_m_vir,count_hod,halo_m_vir,label='SAM: shuffled HOD')
plot_hod(halo_m_vir,count_sam,halo_m_vir,label='SAM')

# Plot the HOD for the hydro
#plot_hod(GrMcrit_fp,count_hod_fp,GrMcrit_fp,label='Hydro: shuffled HOD')
plot_hod(GrMcrit_fp,count_fp,GrMcrit_fp,label='Hydro')
    
# Show both and record
plt.legend()
plt.savefig('figs/HOD.png')
plt.legend()
plt.show()
