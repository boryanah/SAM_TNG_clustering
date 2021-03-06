import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
#from tools.matching import get_match

import plotparams
plotparams.buba()

sim_name = 'TNG300'
#sim_name = 'TNG100'


sam_dir = '/mnt/alan1/boryanah/SAM_subvolumes_'+sim_name+'/'
hydro_dir = "/mnt/gosling1/boryanah/TNG300/"

sample = 'TNG'
#sample = 'SAM'

snap_str = ''
#snap_str = '_55'
#mass_type = 'Crit'
mass_type = 'Mean'

type_dict = {'conc': ["GroupConc_nfw_dm.npy", 'HalopropC_nfw'+snap_str+'.npy', r"$c_{\rm NFW}$"], 'nsub': ["GroupNsubs_dm.npy", "no shit", r"$N_{\rm subs}$"], 'env': ["GroupEnv_dm.npy", 'HalopropEnvironment'+snap_str+'.npy', r"${\rm envir.}$"], 'spin': ["GroupSpin_dm.npy", "HalopropSpin.npy", r"${\rm spin}$"], 'tform': ["no shit", 'Halopropz_Mvir_half'+snap_str+'.npy', r"$z_{\rm half}$"], 'vmax': ["Group_Vmax_dm.npy", 'HalopropVdisk_peak'+snap_str+'.npy', r"$V_{\rm max}$"], 'vani': ["GroupVelAnis_dm_300000.npy", 'no shit', r"$V_{\rm ani}$"], 'mover': ["GroupNsubs_dm.npy", "Group_M_Crit200_dm.npy", r"$M_{\rm FoF}/M_{\rm 200}$"]}#, 'mover': ["GroupMassType_dm.npy", "Group_M_Crit200_dm.npy", r"$M_{\rm FoF}/M_{\rm 200}$"], }

def density_estimation(m1, m2):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    R, _ = stats.spearmanr(m1, m2)
    return X, Y, Z, R

pairs = [('conc', 'vmax'), ('conc','env'), ('conc','nsub'), ('conc','spin'), ('conc','spin'), ('conc','tform'),  ('env','nsub'), ('env','spin'), ('env', 'tform')]
pairs = [('conc', 'vani'), ('env', 'vani')] # TESTING for vani
pairs = [('env', 'mover')] # TESTING for m ratio

for pair in pairs:
    print(pair)
    x_type, y_type = pair
    
    if sample == 'TNG':
        if type_dict[x_type][0] == 'no shit' or type_dict[y_type][0] == 'no shit': continue

        x = np.load(hydro_dir+type_dict[x_type][0])
        #y = np.load(hydro_dir+type_dict[y_type][0]) # og
        #y = np.zeros(len(x)) # TESTING for vani
        #y[:300000] = np.load(hydro_dir+type_dict[y_type][0]) # TESTING for vani
        #y = np.load(hydro_dir+type_dict[y_type][0])[:, 1]/np.load(hydro_dir+type_dict[y_type][1]) # TESTING for m ratio fof
        y = np.load(hydro_dir+type_dict[y_type][0])/np.load(hydro_dir+type_dict[y_type][1]) # TESTING for m ratio n subs
        mass = np.load(hydro_dir+"Group_M_Crit200_dm.npy")*1.e10


    elif sample == 'SAM':
        if type_dict[x_type][1] == 'no shit' or type_dict[y_type][1] == 'no shit': continue
        mass = np.load(sam_dir+'HalopropMvir'+snap_str+'.npy')
        x = np.load(sam_dir+type_dict[x_type][1])
        y = np.load(sam_dir+type_dict[y_type][1])


    lms = np.linspace(12, 13.5, 4) # og
    #lms = np.linspace(12, 13.5, 7)
    #lms = np.linspace(12, 13.4, 5)
    print(lms)

    cs = ['dodgerblue', '#CC6677', 'gray', 'crimson', 'navy', 'black']

    x_max = -1000000.
    y_max = -1000000.
    x_min = 1000000.
    y_min = 1000000.

    fig, ax = plt.subplots(figsize=(9,7))
    for i in range(len(lms)-1):
        lm_max = lms[i+1]
        lm_min = lms[i]
        m_max = 10.**lm_max
        m_min = 10.**lm_min

        choice = (mass < m_max) & (mass > m_min)

        print("number of halos to show = ", np.sum(choice))

        xmin = np.percentile(x[choice], 2.5)
        xmax = np.percentile(x[choice], 97.5)
        ymin = np.percentile(y[choice], 2.5)
        ymax = np.percentile(y[choice], 97.5)

        X, Y, Z, R = density_estimation(x[choice], y[choice])
        print("R = ", R)
        
        z_1 = np.percentile(Z, 70.)
        z_2 = np.percentile(Z, 95.)

        # Add contour lines
        plt.text(0.8, 0.9-i*0.1, r"$r_s = %.2f$"%R, ha='center', va='center', color=cs[i], transform=plt.gca().transAxes)
        plt.plot([], [], c=cs[i], label=r"$\log M = %.1f - %.1f$"%(lm_min, lm_max))
        plt.contour(X, Y, Z, colors=cs[i], levels=[z_1, z_2])

        if xmin < x_min: x_min = xmin
        if ymin < y_min: y_min = ymin
        if xmax > x_max: x_max = xmax
        if ymax > y_max: y_max = ymax #1.2 vani

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    plt.legend(loc='upper left')
    plt.ylabel(type_dict[y_type][-1])
    plt.xlabel(type_dict[x_type][-1])
    plt.savefig("figs/"+x_type+"_"+y_type+"_"+sample+".png")
    plt.show()
    plt.close()
