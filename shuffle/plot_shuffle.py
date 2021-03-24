#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
#import distinct_colours

#num_gals = [1200,6000,12000]
num_gals = [6000,12000,24000]
#type_gals = ['mstar','sfr']
#type_gals = ['mstar','mhalo']
type_gals = ['mstar','mstar_cent']

type_dict = {'mstar': 'M_\\ast$-${\\rm selected', 'mstar_cent': 'M_\\ast$-${\\rm selected', 'mhalo': 'M_{\rm halo}$-${\rm selected', 'sfr': '{\rm SFR}$-${\rm selected', 'sfr_cent': '{\rm SFR}$-${\rm selected \ centrals'}

snap_str = ''
#snap_str = '_55'

line = np.linspace(0,40,3)

bin_centers = np.load("data/bin_centers.npy")

# maybe load average

fig, axes = plt.subplots(len(type_gals),len(num_gals),figsize=(18,9))

color_sam = 'dodgerblue'
color_hydro = '#CC6677'
ls = '-'
k = 0
for i in range(len(type_gals)):
    for j in range(len(num_gals)):
        
        plt.subplot(len(type_gals),len(num_gals),k+1)

        type_gal = type_gals[i]
        num_gal = num_gals[j]
        type_str = type_dict[type_gal]

        if 'cent' in type_gal:
            objs = 'centrals'
        else:
            objs = 'gals.'
        plot_label = r'$ %s, \ %d \ %s}$'%(type_str,num_gal,objs)

        rat_mean_sam = np.load("data/rat_mean_sam_"+str(num_gal)+"_"+type_gal+"_shuff"+snap_str+".npy")
        rat_err_sam = np.load("data/rat_err_sam_"+str(num_gal)+"_"+type_gal+"_shuff"+snap_str+".npy")
        rat_mean_hydro = np.load("data/rat_mean_hydro_"+str(num_gal)+"_"+type_gal+"_shuff"+snap_str+".npy")
        rat_err_hydro = np.load("data/rat_err_hydro_"+str(num_gal)+"_"+type_gal+"_shuff"+snap_str+".npy")

        plt.plot(line,np.ones(len(line)),'k--')

        plt.plot([],[],color=color_sam,ls=ls,label=r'${\rm SAM}$')
        plt.plot([],[],color=color_hydro,ls=ls,label=r'${\rm TNG}$')

        plt.errorbar(bin_centers,rat_mean_sam,yerr=rat_err_sam,color=color_sam,ls='-',alpha=1.,fmt='o',capsize=4)

        plt.errorbar(bin_centers*1.05,rat_mean_hydro,yerr=rat_err_hydro,color=color_hydro,ls='-',alpha=1.,fmt='o',capsize=4)

        plt.xscale('log')
        #plt.yscale('log')
        if k == 0:
            #plt.legend(bbox_to_anchor=(-0.12, 1), loc='upper right',frameon=False,fontsize=18)
            plt.legend(loc='lower right',frameon=False,fontsize=18)
        plt.xlim([0.08,13])
        plt.ylim([0.4,1.5])
        plt.text(0.1,1.3,plot_label)

        if k >= 3:
            plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
        else:
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticks([])
        if k in [0,3]:
            plt.ylabel(r'$\xi(r)_{\rm shuffled}/\xi(r)_{\rm unshuffled}$')
        else:
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticks([])
        k += 1
plt.savefig("figs/shuffle_all"+snap_str+".png")
plt.show()
