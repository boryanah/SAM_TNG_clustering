import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
#import distinct_colours

#secondary_properties = ['env','rvir','conc','vdisp','s2r','spin']
secondary_properties = ['env','rvir','conc','vdiskpeak','mpeak','tform']
#sec_labels = [r'${\rm env.}$',r'$R_{\rm vir}$',r'$c_{\rm NFW}$',r'$V_{\rm disp}$',r'$V_{\rm disp}^2 R$',r'${\rm spin}$']
sec_labels = [r'${\rm env.}$',r'$R_{\rm vir}$',r'$c_{\rm NFW}$',r'$V_{\rm disk, peak}$',r'$M_{\rm peak}$',r'$t_{\rm form.}$']

num_gals = 12000 #6000 # 1200 # 12000
type_gal = 'mstar'#'mstar_cent' #'mhalo' #'mstar'#'sfr'

type_dict = {'mstar': 'M_\\ast$-${\\rm selected', 'mstar_cent': 'M_\ast$-${\\rm selected \ centrals', 'mhalo': 'M_{\\rm halo}$-${\\rm selected', 'sfr': '{\\rm SFR}$-${\\rm selected', 'sfr_cent': '{\\rm SFR}$-${\\rm selected \ centrals'}
type_str = type_dict[type_gal]

line = np.linspace(0,40,3)

#snap_str = ''
snap_str = '_55'

bin_cents = np.load("data/bin_cents.npy")

# maybe load average

fig, axes = plt.subplots(2,3,figsize=(18,9))
#fig.suptitle('%s, %d gals.'%(type_str,num_gals), fontsize=23, y=1.004)

n_sec = len(secondary_properties)

color_sam = 'dodgerblue'
color_hydro = '#CC6677'
ls_top = '-'
ls_bot = '--'

for i in range(n_sec):
    plt.subplot(2,3,i+1)
    
    secondary_property = secondary_properties[i]
    sec_label = sec_labels[i]
    
    shmr_sam_top = np.load("data/shmr_sam_top_"+str(num_gals)+"_"+type_gal+"_"+secondary_property+snap_str+".npy")
    shmr_sam_bot = np.load("data/shmr_sam_bot_"+str(num_gals)+"_"+type_gal+"_"+secondary_property+snap_str+".npy")
    shmr_hydro_top = np.load("data/shmr_hydro_top_"+str(num_gals)+"_"+type_gal+"_"+secondary_property+snap_str+".npy")
    shmr_hydro_bot = np.load("data/shmr_hydro_bot_"+str(num_gals)+"_"+type_gal+"_"+secondary_property+snap_str+".npy")
    
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
        #plt.legend(bbox_to_anchor=(-0.12, 1), loc='upper right',frameon=False,fontsize=18)
        plt.legend(loc='upper left',frameon=False,fontsize=18)
    plt.ylim([1.e-4,0.2])
    plt.xlim([2.e11,1.e15])
    plt.text(1.e14,.05,sec_label)

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
plt.savefig("figs/shmr_param_"+type_gal+snap_str+".png")
plt.show()
