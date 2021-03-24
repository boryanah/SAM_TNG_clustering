#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

#import distinct_colours

num_gals = [6000,12000,24000]
type_gals = ['mstar','sfr']
#type_gals = ['sfr','sfr_cent']
#type_gals = ['mstar','mhalo']
#type_gals = ['mstar','mstar_cent']

type_dict = {'mstar': 'M_\\ast$-${\\rm selected', 'mstar_cent': 'M_\ast$-${\\rm selected \ centrals', 'mhalo': 'M_{\\rm halo}$-${\\rm selected', 'sfr': '{\\rm SFR}$-${\\rm selected', 'sfr_cent': '{\\rm SFR}$-${\\rm selected \ centrals'}

line = np.linspace(0,40,3)

bin_cents = np.load("data/bin_cents.npy")

# maybe load average

fig, axes = plt.subplots(len(type_gals),len(num_gals),figsize=(18,9))

#snap_str = ''
snap_str = '_55'

color_sam = 'dodgerblue'
color_hydro = '#CC6677'
ls = '-'
lw_c = 2.5
lw_s = 1.5
k = 0
for i in range(len(type_gals)):
    for j in range(len(num_gals)):
        
        plt.subplot(len(type_gals),len(num_gals),k+1)

        type_gal = type_gals[i]
        num_gal = num_gals[j]
        type_str = type_dict[type_gal]
        
        plot_label = r'$ %s, \ %d \ gals.}$'%(type_str,num_gal)

        hist_sam = np.load("data/hist_sam_"+str(num_gal)+"_"+type_gal+snap_str+".npy")#
        hist_hydro = np.load("data/hist_hydro_"+str(num_gal)+"_"+type_gal+snap_str+".npy")#
        hist_cents_sam = np.load("data/hist_cents_sam_"+str(num_gal)+"_"+type_gal+snap_str+".npy")#
        hist_cents_hydro = np.load("data/hist_cents_hydro_"+str(num_gal)+"_"+type_gal+snap_str+".npy")#
        plt.plot(line,np.ones(len(line)),'k--')

        plt.plot([],[],color=color_sam,ls=ls,label=r'${\rm SAM}$')
        plt.plot([],[],color=color_hydro,ls=ls,label=r'${\rm TNG}$')
        
        plt.plot(bin_cents,hist_sam-hist_cents_sam,lw=lw_s,ls=ls,color=color_sam)
        plt.plot(bin_cents,hist_hydro-hist_cents_hydro,lw=lw_s,ls=ls,color=color_hydro)
        plt.plot(bin_cents,hist_cents_sam,ls=ls,lw=lw_c,color=color_sam)
        plt.plot(bin_cents,hist_cents_hydro,ls=ls,lw=lw_c,color=color_hydro)

        plt.xscale('log')
        plt.yscale('log')
        if k == 0:
            #plt.legend(bbox_to_anchor=(-0.12, 1), loc='upper right',frameon=False,fontsize=18)
            plt.legend(loc='lower right',frameon=False,fontsize=18)
        if 'SFR' in type_str:
            plt.ylim([1.e-3,100.])
        else:
            plt.ylim([7.e-3,100.])
        plt.xlim([2.e11,1.e15])
        plt.text(4.e11, 18, plot_label)

        if k >= 3:
            plt.xlabel(r'$M_{\rm halo}$')
        else:
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticks([])
        if k in [0,3]:
            plt.ylabel(r'$\langle N_{\rm gal} \rangle$') #
        else:
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticks([])
        k += 1
plt.savefig("figs/hod_all"+snap_str+".png") #
plt.show()
