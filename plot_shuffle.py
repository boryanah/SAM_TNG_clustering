import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
import distinct_colours

secondary_properties = ['env','rvir','conc','conc_desc','vdisp','vdisp_desc','s2r_desc','s2r_asc','spin_desc']
sec_labels = [r'${\rm env. \ (desc.)}$',r'$R_{\rm vir} \ {\rm (desc.)}$',r'$c_{\rm NFW} \ {\rm (mix)}$',r'$c_{\rm NFW} \ {\rm (desc.)}$',r'$V_{\rm disp} \ {\rm (mix)}$',r'$V_{\rm disp} \ {\rm (desc.)}$',r'$V_{\rm disp}^2 R \ {\rm (desc.)}$',r'$V_{\rm disp}^2 R \ {\rm (asc.)}$',r'${\rm spin \ (desc.)}$']
line = np.linspace(0,40,3)


bin_centers = np.load("data_rat/bin_centers.npy")


rat_mean_shuff_sam = np.load("data_rat/rat_mean_sam_shuff.npy")
rat_err_shuff_sam = np.load("data_rat/rat_err_sam_shuff.npy")
    
rat_mean_shuff_hydro = np.load("data_rat/rat_mean_hydro_shuff.npy")
rat_err_shuff_hydro = np.load("data_rat/rat_err_hydro_shuff.npy")


plt.subplots(3,3,figsize=(18,12))

n_sec = len(secondary_properties)

for i in range(n_sec):
    plt.subplot(3,3,i+1)
    
    secondary_property = secondary_properties[i]
    sec_label = sec_labels[i]
    
    rat_mean_sam = np.load("data_rat/rat_mean_sam_"+secondary_property+".npy")
    rat_err_sam = np.load("data_rat/rat_err_sam_"+secondary_property+".npy")
    
    rat_mean_hydro = np.load("data_rat/rat_mean_hydro_"+secondary_property+".npy")
    rat_err_hydro = np.load("data_rat/rat_err_hydro_"+secondary_property+".npy")

    plt.plot(line,np.ones(len(line)),'k--')

    plt.plot(bin_centers,rat_mean_shuff_sam,color='silver',label='SAM shuff.')
    plt.fill_between(bin_centers,rat_mean_shuff_sam-rat_err_shuff_sam,rat_mean_shuff_sam+rat_err_shuff_sam,color='silver',alpha=0.1)
    
    plt.plot(bin_centers,rat_mean_shuff_hydro,color='dimgray',label='Hydro shuff.')
    plt.fill_between(bin_centers,rat_mean_shuff_hydro-rat_err_shuff_hydro,rat_mean_shuff_hydro+rat_err_shuff_hydro,color='dimgray',alpha=0.1)
    
    plt.errorbar(bin_centers,rat_mean_sam,yerr=rat_err_sam,color='dodgerblue',ls='-',label='SAM',alpha=1.,fmt='o',capsize=4)

    plt.errorbar(bin_centers*1.05,rat_mean_hydro,yerr=rat_err_hydro,color='orange',ls='-',label='Hydro',alpha=1.,fmt='o',capsize=4)

    if i >= 6:
        plt.xlabel(r'$r [{\rm Mpc}/h]$')
    else:
        plt.gca().axes.xaxis.set_ticklabels([])
    if i in [0,3,6]:
        plt.ylabel(r'$\xi(r)_{\rm model}/\xi(r)_{\rm TNG300}$')
    else:
        plt.gca().axes.yaxis.set_ticklabels([])
    plt.xscale('log')
    if i == 0:
        plt.legend(bbox_to_anchor=(-0.12, 1), loc='upper right',frameon=False,fontsize=20)
    plt.xlim([0.08,13])
    plt.ylim([0.4,1.5])
    plt.text(0.2,0.5,sec_label)
plt.savefig("figs/mock_ratio_all.png")
#plt.show()
