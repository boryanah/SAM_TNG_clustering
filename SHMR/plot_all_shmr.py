import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
#import distinct_colours

secondary_properties = ['env','rvir','conc','vdisp','s2r','spin']
sec_labels = [r'${\rm env.}$',r'$R_{\rm vir}$',r'$c_{\rm NFW}$',r'$V_{\rm disp}$',r'$V_{\rm disp}^2 R$',r'${\rm spin}$']

num_gals = 6000 #6000 # 1200 # 12000
type_gal = 'mhalo'#'mhalo' #'mstar'#'sfr'

type_dict = {'mstar': 'Mass-selected', 'sfr': 'SFR-selected', 'mhalo': 'HaloMass-selected',}
type_str = type_dict[type_gal]

line = np.linspace(0,40,3)

bin_cents = np.load("data_shmr/bin_cents.npy")

# maybe load average

fig, axes = plt.subplots(2,3,figsize=(18,9))
fig.suptitle('%s, %d gals.'%(type_str,num_gals), fontsize=23, y=1.004)

n_sec = len(secondary_properties)

color_sam = 'dodgerblue'
color_hydro = '#CC6677'
ls_top = '-'
ls_bot = '--'

for i in range(n_sec):
    plt.subplot(2,3,i+1)
    
    secondary_property = secondary_properties[i]
    sec_label = sec_labels[i]
    
    shmr_sam_top = np.load("data_shmr/shmr_sam_top_"+str(num_gals)+"_"+type_gal+"_"+secondary_property+".npy")
    shmr_sam_bot = np.load("data_shmr/shmr_sam_bot_"+str(num_gals)+"_"+type_gal+"_"+secondary_property+".npy")
    shmr_hydro_top = np.load("data_shmr/shmr_hydro_top_"+str(num_gals)+"_"+type_gal+"_"+secondary_property+".npy")
    shmr_hydro_bot = np.load("data_shmr/shmr_hydro_bot_"+str(num_gals)+"_"+type_gal+"_"+secondary_property+".npy")
    
    plt.plot(line,np.ones(len(line)),'k--')

    plt.plot([],[],color=color_sam,ls=ls_top,label=r'SAM top 25\%')
    plt.plot([],[],color=color_hydro,ls=ls_top,label=r'TNG top 25\%')
    plt.plot([],[],color=color_sam,ls=ls_bot,label=r'SAM bottom 25\%')
    plt.plot([],[],color=color_hydro,ls=ls_bot,label=r'TNG bottom 25\%')

    plt.plot(bin_cents,shmr_sam_top,ls=ls_top,color=color_sam)
    plt.plot(bin_cents,shmr_sam_bot,ls=ls_bot,color=color_sam)

    plt.plot(bin_cents,shmr_hydro_top,ls=ls_top,color=color_hydro)
    plt.plot(bin_cents,shmr_hydro_bot,ls=ls_bot,color=color_hydro)
    
    plt.xscale('log')
    plt.yscale('log')
    if i == 0:
        plt.legend(bbox_to_anchor=(-0.12, 1), loc='upper right',frameon=False,fontsize=18)
    plt.ylim([1.e-4,0.1])
    plt.xlim([1.e10,1.e15])
    plt.text(1.e12,.001,sec_label)

    if i >= 3:
        plt.xlabel(r'$M_{\rm halo}$')
    else:
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.xaxis.set_ticks([])
    if i in [0,3]:
        plt.ylabel(r'$\langle N_{\rm gal} \rangle$')
    else:
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticks([])
plt.savefig("figs/SHMR_all.png")
#plt.show()
