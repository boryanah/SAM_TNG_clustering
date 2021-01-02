import matplotlib.pyplot as plt
import numpy as np
import plotparams # for making plots pretty, but comment out
plotparams.buba() # same
import Corrfunc # comment out since you're not using it

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
                # TESTING


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
    N_bins = 31
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
def get_shuff_counts(count_halo,group_mass,record_relative=False):
    # define mass bins
    log_min = 10.
    log_max = 15.
    N_bins = 21
    bin_edges = np.linspace(log_min,log_max,N_bins)
    bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

    if record_relative:
        assert count_halo.shape[1] == 2, "you sure?"

    N_halos = len(count_halo)
    #count_halo_hod = np.zeros(N_halos,dtype=int)
    # TESTING!!! copies
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

# Hubble constant, snapshot number (99 is last snapshot, i.e. z = 0), number of galaxies
h = 0.6774
k = 99
num_gals = 6000
Lbox = 75.
n_gal = num_gals/Lbox**3.
record_relative = 1

##########################################
#####               SAM              ##### 
##########################################

# Loading data for the SAM
#SAM_dir = '/mnt/gosling1/boryanah/SAM/TNG100-1-SAM/'
SAM_dir = '/mnt/gosling1/boryanah/SAM/TNG100-1-SAM-Recalibrated/'
hosthaloid = np.load(SAM_dir+'hosthaloid_%d.npy'%k)
mhalo = np.load(SAM_dir+'mhalo_%d.npy'%k)
mstar = np.load(SAM_dir+'mstar_%d.npy'%k)
sat_type = np.load(SAM_dir+'sat_type_%d.npy'%k)
x_position = np.load(SAM_dir+'x_position_%d.npy'%k)
y_position = np.load(SAM_dir+'y_position_%d.npy'%k)
z_position = np.load(SAM_dir+'z_position_%d.npy'%k)
N_halos_sam = len(sat_type)

# combine the halo positions into one file
xyz_position = np.vstack((x_position,y_position,z_position)).T
xyz_position[xyz_position > Lbox] -= Lbox
xyz_position[xyz_position < 0.] += Lbox 

# order the halos in terms of their stellar masses (notice that I am not cutting them at top 6000, but just reordering)
inds_top = np.argsort(mstar)[::-1]
mhalo_top = mhalo[inds_top]
sat_top = sat_type[inds_top]
hosthaloid_top = hosthaloid[inds_top]
xyz_top = xyz_position[inds_top]

# Count the number of galaxies per halo in the SAM
# start commenting out here for skipping the steps below if you have already run this ones and saved the arrays in data/
'''
def get_count_sam(N_halos,num_gals,mhalo_top,hosthaloid_top,sat_top,xyz_top,record_relative=False):
    # create an empty array with the counts per halo for all of the SAM halos (later we will filter those to be only the big halos, but for convenience not doing so now)
    count = np.zeros((N_halos),dtype=np.int64)
    # copy the sat type array because we will be modifying it
    sat = sat_top.copy()
    
    # TODO perhaps only do it for the unique dudes
    
    if record_relative:
        relative = np.zeros((num_gals,3))
        nstart = np.zeros(N_halos,dtype=int)-1
        
        cumulative = 0

    hosthaloid_gals = hosthaloid_top[:num_gals]
    un_hosthaloid_gals = np.unique(hosthaloid_gals)
    num_unique = len(un_hosthaloid_gals)
    print("number of unique haloids (this is almost equivalent to number of centrals since we assume one central per halo)",num_unique)

    # wtf counts the cases where there is no central galaxy for that hosthaloid
    wtf = 0
    for i in range(num_unique):
        if i % 100 == 0: print(i)

        # for this galaxy (notice that the initial array is ordered in terms of stellar mass so top 6000 objects are truly the gals):
        # for this galaxy which of the halos share its hosthaloid
        choice = hosthaloid_top == un_hosthaloid_gals[i]
        
        # how many of the halos sharing it are galaxies (i.e. are in the top 6000?) 
        n_gal = np.sum(choice[:num_gals])
        # how many of the halos sharing the hosthaloid are centrals of the large halo (hopefully just 1 -- otherwise problems) 
        center_choice = sat_top[choice] == 0

        # for these cases when there is no central halo (i.e. no big halo that encompasses the little ones, idk why there isn't), we will instead select the halo with highest halo mass sharing the same hosthaloid and call that the central halo (this is my strike of creativity). 
        if np.sum(center_choice) == 0:
            # find the most massive halo sharing the same hosthaloid
            i_max = np.argmax(mhalo_top[choice])

            # select the new halo center
            tmp = sat[choice] 
            tmp[i_max] = 0
            sat[choice] = tmp

            xyz_center = (xyz_top[choice])[tmp == 0]

            # set the counts for this new halo center (which we select to be the host)
            tmp = count[choice] 
            tmp[i_max] = n_gal
            count[choice] =  tmp
            
            wtf += 1
            

        else:
            # for the non-weird case make sure we have exactly one central
            assert np.sum(center_choice) == 1, "should be just one central"

            # set the counts for this halo center (which is a host)
            tmp = count[choice] 
            tmp[center_choice] = n_gal
            count[choice] =  tmp
            xyz_center = (xyz_top[choice])[center_choice]
        
        if record_relative:
            # doesn't matter since we only select central halos at the end
            inds_choice = np.arange(xyz_top.shape[0])[choice]
            relative[cumulative:cumulative+n_gal] = (xyz_top[inds_choice[inds_choice<num_gals]]) - xyz_center
            nstart[choice] = cumulative
            
            cumulative += n_gal

    # how many weird cases with no central
    print("no central exists for the halo = ",wtf)


    # choose only the halos which are centrals (which is to say they are big) and save their occupation number, positions and mhalo
    choose = sat == 0
    count = count[choose]
    mhalo = mhalo_top[choose]
    xyz = xyz_top[choose]

    if record_relative:
        print("number of galaxies considered = ",cumulative)

        nstart = nstart[choose]
        
        return count, mhalo, xyz, sat, relative, nstart

    return count, mhalo, xyz, sat

# return the galaxy count per halo array, halo mass of the central halo and sat type array (notice it is also sorted by stellar mass)
#count_sam, mhalo_sam, xyz_sam, sat_sam = get_count_sam(N_halos_sam,num_gals,mhalo_top,hosthaloid_top,sat_top,xyz_top)
# TESTING
count_sam, mhalo_sam, xyz_sam, sat_sam, rel_pos_gals_sam, nstart_sam = get_count_sam(N_halos_sam,num_gals,mhalo_top,hosthaloid_top,sat_top,xyz_top,record_relative=record_relative)

# save the big halos occupation number, positions and mhalo; also the newly made sat_type of the entire array so we can isolate future properties in the future after sorting by stellar mass and applying sat_type == 0
np.save("data/count_sam.npy",count_sam)
np.save("data/xyz_sam.npy",xyz_sam)
np.save("data/mhalo_sam.npy",mhalo_sam)
np.save("data/sat_sam.npy",sat_sam)
if record_relative:
    np.save("data/rel_pos_gals_sam.npy",rel_pos_gals_sam)
    np.save("data/nstart_sam.npy",nstart_sam)

print(np.sum(count_sam))

# end commenting out here for skipping the steps above
'''

# load constructed counts per halo (i.e. constructed HOD)
count_sam = np.load("data/count_sam.npy")
mhalo_sam = np.load("data/mhalo_sam.npy")
xyz_sam = np.load("data/xyz_sam.npy")
sat_sam = np.load("data/sat_sam.npy")

if record_relative:
    rel_pos_gals_sam = np.load("data/rel_pos_gals_sam.npy")
    nstart_sam = np.load("data/nstart_sam.npy")

# version 1 -- random draws for satellites and centrals (if you remember this, but if not, don't worry)
# to get the counts for the centrals we assume that every halo with at least one galaxy has a central
#count_cents_sam = count_sam.copy()
#count_cents_sam[count_sam > 0] = 1
#hist_sam, bin_cents = get_hod(mhalo_sam,count_sam,mhalo_sam)
#hist_cents_sam, bin_cents = get_hod(mhalo_sam,count_cents_sam,mhalo_sam)
#hist_sats_sam = hist_sam - hist_cents_sam
#count_hod = get_hod_counts(hist_sam,hist_cents_sam,hist_sats_sam,N_halos_sam,mhalo_sam)
# version 2 -- random shuffling in halo mass bins
#count_hod = get_shuff_counts(count_sam,mhalo_sam)
# version 3 -- putting in centers
count_nstart_sam = np.vstack((count_sam,nstart_sam)).T
count_hod, nstart_hod = get_shuff_counts(count_nstart_sam,mhalo_sam,record_relative=record_relative)

show_hod = 1
if show_hod:
    # Plot the HOD
    plt.figure(2,figsize=(8,7))
    plot_hod(mhalo_sam,count_hod,mhalo_sam,label='SAM: shuffled HOD')
    plot_hod(mhalo_sam,count_sam,mhalo_sam,label='SAM: true HOD')
    
    # Show both and record
    plt.legend()
    plt.savefig('figs/HOD.png')
    #plt.show()

# our halo positions are ordered in terms of stellar mass so simply select the top 6000 -> those are our true positions of the galaxies
xyz_true = xyz_top[:num_gals]
# these are the weigths which are all 1s (since we are computing the correlation function of all galaxies rather than placing them at the center)
w_true = np.ones(num_gals,dtype=xyz_true.dtype)

# here we select out of our newly constructed array with counts per halo only those halos which have galaxies and put the galaxies in the center (weights are not all ones)
xyz_cents = xyz_sam[count_sam>0]
w_cents = count_sam[count_sam>0].astype(xyz_cents.dtype)

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

# version 1 -- put in center
# this is the shuffled HOD where again we put the galaxies at the center of the halos
#xyz_hod = xyz_sam[count_hod > 0]
#w_hod = count_hod[count_hod > 0].astype(xyz_hod.dtype)
# version 2 -- put relative positions
xyz_hod, w_hod = get_xyz_w(count_hod,nstart_hod,xyz_sam,rel_pos_gals_sam,xyz_cents.dtype,Lbox)

# compute the ratio of the corr func of the constructed HOD to the true SAM with error bars and plot it
plt.figure(1,figsize=(8,7))
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_cents,w_cents,Lbox)
plt.plot(bin_centers,np.ones(len(bin_centers)),'k--')
# this curve should be consistent with 1 on large scales
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='orange',ls='-',label='SAM: halo cents/true',alpha=1.,fmt='o',capsize=4)
    

# compute the ratio of the shuffled galaxies to the true SAM distribution with error bars and plot it
plt.figure(1,figsize=(8,7))
Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='-',label='SAM: shuffled/true',alpha=1.,fmt='o',capsize=4)

# display the ratios
plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.5,1.5])
#plt.savefig("figs/mock_ratio.png")


##########################################
#####               HYDRO            ##### 
##########################################

# Loading data for the hydro -- you can skip for now
hydro_dir = '/mnt/gosling1/boryanah/TNG100/'
SubGrNr_fp = np.load(hydro_dir+'SubhaloGrNr_fp.npy')
GrMcrit_fp = np.load(hydro_dir+'Group_M_Crit200_fp.npy')*1.e10
GroupPos_fp = np.load(hydro_dir+'GroupPos_fp.npy')/1000.
SubhaloPos_fp = np.load(hydro_dir+'SubhaloPos_fp.npy')/1000.
SubMstar_fp = np.load(hydro_dir+'SubhaloMassType_fp.npy')[:,4]*1.e10

# total number of halos
N_halos_hydro = len(GrMcrit_fp)

# Find the indices of the top galaxies
inds_top = np.argsort(SubMstar_fp)[::-1][:num_gals]

# this function returns the HOD, bin centers of the HOD and most importantly galaxy counts per halo for subhalos with indices inds_top
def get_hist_count(inds_top,record_relative=False):
    # Find the halo ID's of their halo parents in the original GrMcrit_fp array
    hosts_top = SubGrNr_fp[inds_top]

    pos_top = SubhaloPos_fp[inds_top]
    # Find the masses of their halo parents from the original GrMcrit_fp array
    masses_top = GrMcrit_fp[hosts_top]
    # Knowing the ID's of the relevant halos (i.e. those who are hosting a galaxy),
    # tell me which ID's are unique, what indices in the hosts_top array these
    # unique ID's correspond to and how many times they each repeat
    hosts, inds, gal_counts_hydro = np.unique(hosts_top,return_index=True,return_inverse=False, return_counts=True)

    # get unique masses of hosts to compute the HOD (hist_hydro)
    host_masses_hydro = masses_top[inds]
    hist_hydro, bin_cents = get_hod(host_masses_hydro,gal_counts_hydro,GrMcrit_fp)

    # get the galaxy counts per halo
    count_halo_fp = np.zeros(N_halos_hydro,dtype=int)
    count_halo_fp[hosts] += gal_counts_hydro

    if record_relative:
        nstart_halo_fp = np.zeros(N_halos_hydro,dtype=int)-1
        rel_pos_gals_fp = np.zeros((len(inds_top),3))
        cumulative = 0
        for i in range(len(hosts)):
            group_cen = GroupPos_fp[hosts[i]]
            n_gal = gal_counts_hydro[i]

            gal_sel = hosts[i] == hosts_top
            assert np.sum(gal_sel) == n_gal, "you must have messed up"
            rel_pos_gals_fp[cumulative:cumulative+n_gal] = pos_top[gal_sel]-group_cen
            nstart_halo_fp[hosts[i]] = cumulative
            
            cumulative += n_gal
        print("total number of galaxies = ",cumulative)
        return hist_hydro, bin_cents, count_halo_fp, nstart_halo_fp, rel_pos_gals_fp
    
    return hist_hydro, bin_cents, count_halo_fp

# total histogram of sats + centrals and galaxy counts per halo
#hist_hydro, bin_cents, count_halo_fp = get_hist_count(inds_top,record_relative=record_relative)
# TESTING
hist_hydro, bin_cents, count_halo_fp, nstart_halo_fp, rel_pos_gals_fp = get_hist_count(inds_top,record_relative=record_relative)

# version 1
# get parent indices of the centrals and their subhalo indices in the original array
#unique_sub_grnr, firsts = np.unique(SubGrNr_fp,return_index=True)
#inds_firsts_gals = np.intersect1d(inds_top,firsts)
# central histogram
#hist_hydro_cents, bin_cents, count_halo_sat_fp = get_hist_count(inds_firsts_gals)
#hist_hydro_sats = hist_hydro-hist_hydro_cents
#count_hod_fp = get_hod_counts(hist_hydro,hist_hydro_cents,hist_hydro_sats,N_halos_hydro,GrMcrit_fp)
# version 2
#count_hod_fp = get_shuff_counts(count_halo_fp,GrMcrit_fp)
# version 3 -- putting in centers
count_nstart_fp = np.vstack((count_halo_fp,nstart_halo_fp)).T
count_hod_fp, nstart_hod_fp = get_shuff_counts(count_nstart_fp,GrMcrit_fp,record_relative=record_relative)

if show_hod:
    plt.figure(2,figsize=(8,7))
    # Plot the HOD for the hydro
    plot_hod(GrMcrit_fp,count_hod_fp,GrMcrit_fp,label='Hydro: shuffled HOD')
    plot_hod(GrMcrit_fp,count_halo_fp,GrMcrit_fp,label='Hydro: true HOD')
    
    # Show both and record
    plt.legend()
    plt.savefig('figs/HOD.png')

# select the galaxies positions and weights
xyz_true = SubhaloPos_fp[inds_top]
w_true = np.ones(xyz_true.shape[0],dtype=xyz_true.dtype)
    
# here we select out of our newly constructed array with counts per halo only those halos which have galaxies and put the galaxies in the center
xyz_cents = GroupPos_fp[count_halo_fp > 0]
w_cents = count_halo_fp[count_halo_fp > 0].astype(xyz_cents.dtype)

# version 1 -- put in centers of halos
# this is the shuffled HOD where again we put the galaxies at the center of the halos
#xyz_hod = GroupPos_fp[count_hod_fp > 0]
#w_hod = count_hod_fp[count_hod_fp > 0].astype(xyz_hod.dtype)
# version 2 -- put relative positions
xyz_hod, w_hod = get_xyz_w(count_hod_fp,nstart_hod_fp,GroupPos_fp,rel_pos_gals_fp,xyz_cents.dtype,Lbox)

plt.figure(1,figsize=(8,7))

Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_cents,w_cents,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='orange',ls='--',label='Hydro: halo cents/true',alpha=1.,fmt='o',capsize=4)

Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='--',label='Hydro: shuffled/true',alpha=1.,fmt='o',capsize=4)

plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.,1.5])
plt.savefig("figs/mock_ratio.png")
plt.show()
