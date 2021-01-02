import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
#import distinct_colours

num_gals = [1200,6000,12000]
#type_gals = ['mstar','sfr']
type_gals = ['mstar','mhalo']

type_dict = {'mstar': 'Mass-selected','mhalo': 'HaloMass-selected', 'sfr': 'SFR-selected'}

line = np.linspace(0,40,3)

bin_cents = np.load("data_hod/bin_cents.npy")

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
        
        plot_label = '%s, %d gals.'%(type_str,num_gal)

        hist_sam = np.load("data_hod/hist_sam_"+str(num_gal)+"_"+type_gal+".npy")#
        hist_hydro = np.load("data_hod/hist_hydro_"+str(num_gal)+"_"+type_gal+".npy")#

        plt.plot(line,np.ones(len(line)),'k--')

        plt.plot([],[],color=color_sam,ls=ls,label=r'SAM')
        plt.plot([],[],color=color_hydro,ls=ls,label=r'TNG')

        plt.plot(bin_cents,hist_sam,ls=ls,color=color_sam)
        plt.plot(bin_cents,hist_hydro,ls=ls,color=color_hydro)

        plt.xscale('log')
        plt.yscale('log')
        if k == 0:
            plt.legend(bbox_to_anchor=(-0.12, 1), loc='upper right',frameon=False,fontsize=18)
        plt.ylim([1.e-2,100.]) #
        plt.xlim([1.e10,1.e15])
        plt.text(1.e11,.014,plot_label)

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
plt.savefig("figs/HOD.png") #
#plt.show()
