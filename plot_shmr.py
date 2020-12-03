import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
#import distinct_colours

num_gals = [1200,6000,12000]
type_gals = ['mstar','sfr']

type_dict = {'mstar': 'Mass-selected', 'sfr': 'SFR-selected'}

line = np.linspace(0,40,3)

bin_cents = np.load("data_shmr/bin_cents.npy")

# maybe load average

fig, axes = plt.subplots(len(type_gals),len(num_gals),figsize=(18,9))

color_sam = 'dodgerblue'
color_hydro = '#CC6677'
ls = '-'
k = 0
for i in range(len(type_gals)):
    for j in range(len(num_gals)):
        
        plt.subplot(len(type_gals),len(num_gals),k+1)
        k += 1
        type_gal = type_gals[i]
        num_gal = num_gals[j]
        type_str = type_dict[type_gal]
        
        plt.title('%s, %d gals.'%(type_str,num_gals), fontsize=23, y=0.995)

        shmr_sam = np.load("data_shmr/shmr_sam_"+str(num_gal)+"_"+type_gal+".npy")#
        shmr_hydro = np.load("data_shmr/shmr_hydro_"+str(num_gal)+"_"+type_gal+".npy")#

        plt.plot(line,np.ones(len(line)),'k--')

        plt.plot([],[],color=color_sam,ls=ls,label=r'SAM')
        plt.plot([],[],color=color_hydro,ls=ls,label=r'TNG')

        plt.plot(bin_cents,shmr_sam,ls=ls,color=color_sam)
        plt.plot(bin_cents,shmr_hydro,ls=ls,color=color_hydro)

        plt.xscale('log')
        plt.yscale('log')
        if i == 0:
            plt.legend(bbox_to_anchor=(-0.12, 1), loc='upper right',frameon=False,fontsize=18)
        plt.ylim([1.e-4,0.1]) #
        plt.xlim([1.e10,1.e15])
        plt.text(1.e12,.001,plot_label)

        if i >= 3:
            plt.xlabel(r'$M_{\rm halo}$')
        else:
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticks([])
        if i in [0,3]:
            plt.ylabel(r'$\langle M_{\rm \ast}/M_{\rm halo} \rangle$') #
        else:
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticks([])
plt.savefig("figs/SHMR.png") #
#plt.show()
