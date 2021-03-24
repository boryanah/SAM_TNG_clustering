import numpy as np
import matplotlib.pyplot as plt
import Corrfunc

def get_hod(masses,counts,other_group_mass=None):
    # number of bin edges
    n_bins = 31
    # bin edges (btw Delta = np.log10(bins[1])-np.log10(bins[0]) in log space)
    bins = np.logspace(10.,15.,n_bins)
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])
    # This histogram tells you how many halos there are in each bin inerval (n_bins-1)
    if other_group_mass is not None:
        
        i_sort = np.argsort(masses)[::-1]
        counts_sorted = counts[i_sort]
        i_sort_other = np.argsort(other_group_mass)[::-1]
        other_group_mass_sorted = other_group_mass[i_sort_other]
        N_min = np.min([len(masses), len(other_group_mass)])
        hist_weighted, edges = np.histogram(other_group_mass_sorted[:N_min],bins=bins,weights=counts_sorted[:N_min])
        hist_norm, edges = np.histogram(other_group_mass_sorted[:N_min],bins=bins)
        hist_hod = hist_weighted/hist_norm
    else:
        hist_norm, edges = np.histogram(masses,bins=bins)
        hist_weighted, edges = np.histogram(masses,bins=bins,weights=counts)
        hist_hod = hist_weighted/hist_norm

    return hist_hod, bin_cents

def get_hod_prop(masses,counts,halo_prop):
    # number of bin edges
    n_bins = 31
    # bin edges (btw Delta = np.log10(bins[1])-np.log10(bins[0]) in log space)
    bins = np.logspace(10.,15.,n_bins)
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])
    # This histogram tells you how many halos there are in each bin inerval (n_bins-1)
    
    hist_hod_top = np.zeros(len(bin_cents))
    hist_hod_bot = np.zeros(len(bin_cents))
    
    for i in range(len(bin_cents)):
        M_lo = bins[i]
        M_hi = bins[i+1]

        mass_sel = (M_lo <= masses) & (M_hi > masses)

        if np.sum(mass_sel) == 0: continue

        masses_sel = masses[mass_sel]
        counts_sel = counts[mass_sel]
        if halo_prop is not None:
            halo_prop_sel = halo_prop[mass_sel]
        
            prop_top = np.percentile(halo_prop_sel,75)
            prop_bot = np.percentile(halo_prop_sel,25)
        
            choice_top = halo_prop_sel >= prop_top
            choice_bot = halo_prop_sel <= prop_bot

            if np.sum(choice_top) == 0: continue
            if np.sum(choice_bot) == 0: continue
        
            hist_hod_top[i] = np.sum(counts_sel[choice_top])/len(counts_sel[choice_top])
            hist_hod_bot[i] = np.sum(counts_sel[choice_bot])/len(counts_sel[choice_bot])
        else:
            
            hist_hod_top[i] = np.sum(counts_sel)/len(counts_sel)
            hist_hod_bot[i] = np.sum(counts_sel)/len(counts_sel)
            
    return hist_hod_top, hist_hod_bot, bin_cents

def plot_hod(hist_hod, bin_cents, label):
    # Plot the derived HOD shape
    plt.plot(bin_cents,hist_hod,label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\langle N_{\rm gal} \rangle$')
    plt.xlabel(r'$M_{\rm halo}$')
    #plt.text(1.e13,0.05,r'$n_{\rm gal} = %.4f$'%n_gal)
    plt.ylim([1.e-2,100.])
    plt.xlim([1.e10,1.e15])

def plot_hod_prop(hist_hod_top, hist_hod_bot, bin_cents, label):
    # Plot the derived HOD shape
    plt.plot(bin_cents,hist_hod_top,ls='--',label=label)
    plt.plot(bin_cents,hist_hod_bot,ls=':')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\langle N_{\rm gal} \rangle$')
    plt.xlabel(r'$M_{\rm halo}$')
    #plt.text(1.e13,0.05,r'$n_{\rm gal} = %.4f$'%n_gal)
    plt.ylim([1.e-2,100.])
    plt.xlim([1.e10,1.e15])

def plot_hmf(masses,label):
    # number of bin edges
    n_bins = 31
    bins = np.logspace(8.,15.,n_bins)
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])

    hist, edges = np.histogram(masses,bins=bins)

    plt.plot(bin_cents,hist,label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$N_{\rm halo}$')
    plt.xlabel(r'$M_{\rm halo} [M_\odot/h]$')

def get_hmf(masses):
    # number of bin edges
    n_bins = 31
    bins = np.logspace(8.,15.,n_bins)
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])

    hist, edges = np.histogram(masses,bins=bins)

    return hist, bin_cents
    
def get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox):
    
    # bins for the correlation function
    N_bin = 21
    bins = np.logspace(-1.,1.,N_bin)
    bin_centers = (bins[:-1] + bins[1:])/2.

    # dimensions for jackknifing
    N_dim = 3

    true_max = xyz_true.max()
    true_min = xyz_true.min()
    hod_max = xyz_hod.max()
    hod_min = xyz_hod.min()
    
    print("true max = ", true_max)
    print("true min = ", true_min)
    print("hod max = ", hod_max)
    print("hod min = ", hod_min)
    if true_max > Lbox or true_min < 0. or hod_max > Lbox or hod_min < 0.:
        print("NOTE: we are using UNFAIR methods")
        xyz_true = xyz_true % Lbox
        xyz_hod = xyz_hod % Lbox

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

def get_shuff_counts(count_halo,group_mass,record_relative=False,order_by=None,order_type='desc'):
    '''
    this function shuffles the number of total galaxies, for each mass bin and returns the number of galaxies living in each halo count_halo_hod
    '''
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


def get_hist_count(inds_top, hosts_top, sub_pos, group_mass, group_pos, Lbox, record_relative=False, group_rad=None, other_group_mass=None):
    '''
    this function returns the HOD, bin centers of the HOD and most importantly galaxy counts per halo for subhalos with indices inds_top
    '''

    pos_top = sub_pos[inds_top]
    # Find the masses of their halo parents from the original group_mass array
    if other_group_mass is not None:
        # tuks
        pass
    
    masses_top = group_mass[hosts_top]
    # Knowing the ID's of the relevant halos (i.e. those who are hosting a galaxy),
    # tell me which ID's are unique, what indices in the hosts_top array these
    # unique ID's correspond to and how many times they each repeat
    hosts, inds, gal_counts = np.unique(hosts_top,return_index=True,return_counts=True)

    # get unique masses of hosts to compute the HOD (hist)
    host_masses = masses_top[inds]
    hist, bin_cents = get_hod(host_masses,gal_counts)

    # number of halos
    N_halos = len(group_mass)
    
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

            if group_rad is not None:
                #print(rel_pos_gals_halo)
                #print(group_rad[hosts[i]])
                #print("---------------")
                rel_pos_gals_halo[cumulative:cumulative+n_gal] /= group_rad[hosts[i]]
                
            cumulative += n_gal
            
        print("total number of galaxies = ",cumulative)
        dx = rel_pos_gals_halo[:,0]
        dy = rel_pos_gals_halo[:,1]
        dz = rel_pos_gals_halo[:,2]
        print((np.min(rel_pos_gals_halo)))
        print((np.max(rel_pos_gals_halo)))

        dx = np.where(np.abs(dx) > Lbox/2.,-np.sign(dx)*np.abs(np.abs(dx)-Lbox),dx)
        dy = np.where(np.abs(dy) > Lbox/2.,-np.sign(dy)*np.abs(np.abs(dy)-Lbox),dy)
        dz = np.where(np.abs(dz) > Lbox/2.,-np.sign(dz)*np.abs(np.abs(dz)-Lbox),dz)
        rel_pos_gals_halo[:,0] = dx
        rel_pos_gals_halo[:,1] = dy
        rel_pos_gals_halo[:,2] = dz
        print((np.min(rel_pos_gals_halo)))
        print((np.max(rel_pos_gals_halo)))
        '''
        print(np.sum(np.abs(rel_pos_gals_halo)))
        print((np.min(rel_pos_gals_halo)))
        print((np.max(rel_pos_gals_halo)))
        print(np.sum(count_halo))
        print(nstart_halo[:100])
        print(count_halo[:100])
        '''

        #if group_rad is not None:
        #    return hist, bin_cents, count_halo, nstart_halo, rel_pos_norm
        
        return hist, bin_cents, count_halo, nstart_halo, rel_pos_gals_halo

    
    return hist, bin_cents, count_halo

def get_xyz_w(count_sel, nstart_sel, xyz_all, rel_pos_gals, dtype, Lbox):
    '''
    returns the xyz position and the weights; also works if feeding relative positions
    '''
    
    xyz_cents_sel = xyz_all[count_sel > 0]
    nstart_sel = nstart_sel[count_sel > 0]
    count_sel = count_sel[count_sel > 0]

    num_gals = np.sum(count_sel)
    
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

def get_from_sub_prop(sub_prop,sub_grnr,n_halo):
    gr_prop = np.zeros(n_halo)
    # gives the unique indices of first occurrence ordered
    unique, inds = np.unique(sub_grnr,return_index=True)
    gr_prop[unique] = sub_prop[inds]
    return gr_prop


def get_shmr(inds_top, haloid, mstar, masses):

    mstar_top = mstar[inds_top]
    mhalo_top = masses[haloid]
    ratio_top = mstar_top/mhalo_top

    n_bins = 31
    # bin edges (btw Delta = np.log10(bins[1])-np.log10(bins[0]) in log space)
    bins = np.logspace(10.,15.,n_bins)
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])
    
    hist_norm, edges = np.histogram(masses,bins=bins)
    hist_weighted, edges = np.histogram(mhalo_top,bins=bins,weights=ratio_top)
    shmr_ratio = hist_weighted/hist_norm
    
    return shmr_ratio, bin_cents

def get_shmr_prop(inds_top, haloid, mstar, masses, halo_prop):

    mstar_top = mstar[inds_top]
    mhalo_top = masses[haloid]
    ratio_top = mstar_top/mhalo_top

    ratios = np.zeros(len(masses))
    ratios[haloid] = ratio_top
    
    n_bins = 31
    # bin edges (btw Delta = np.log10(bins[1])-np.log10(bins[0]) in log space)
    bins = np.logspace(10.,15.,n_bins)
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])

    shmr_top = np.zeros(len(bin_cents))
    shmr_bot = np.zeros(len(bin_cents))
    
    for i in range(len(bin_cents)):
        M_lo = bins[i]
        M_hi = bins[i+1]

        mass_sel = (M_lo <= masses) & (M_hi > masses)

        if np.sum(mass_sel) == 0: continue

        masses_sel = masses[mass_sel]
        ratios_sel = ratios[mass_sel]
        if halo_prop is not None:
            halo_prop_sel = halo_prop[mass_sel]
        
            prop_top = np.percentile(halo_prop_sel,75)
            prop_bot = np.percentile(halo_prop_sel,25)
        
            choice_top = halo_prop_sel >= prop_top
            choice_bot = halo_prop_sel <= prop_bot

            if np.sum(choice_top) == 0: continue
            if np.sum(choice_bot) == 0: continue
        
            shmr_top[i] = np.sum(ratios_sel[choice_top])/len(ratios_sel[choice_top])
            shmr_bot[i] = np.sum(ratios_sel[choice_bot])/len(ratios_sel[choice_bot])
        else:
            
            shmr_top[i] = np.sum(ratios_sel)/len(ratios_sel)
            shmr_bot[i] = np.sum(ratios_sel)/len(ratios_sel)
    
    return shmr_top, shmr_bot, bin_cents

def plot_shmr_prop(shmr_top, shmr_bot, bin_cents, label):
    # Plot the derived HOD shape
    plt.plot(bin_cents,shmr_top,ls='--',label=label)
    plt.plot(bin_cents,shmr_bot,ls=':')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$M_\ast/M_{\rm halo}$')
    plt.xlabel(r'$M_{\rm halo}$')
    plt.ylim([1.e-5,0.1])
    plt.xlim([1.e10,1.e15])

def plot_shmr(shmr_rat, bin_cents, label):
    # Plot the derived HOD shape
    plt.plot(bin_cents,shmr_rat,ls='-',label=label)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$M_\ast/M_{\rm halo}$')
    plt.xlabel(r'$M_{\rm halo}$')
    plt.ylim([1.e-5,0.1])
    plt.xlim([1.e10,1.e15])
