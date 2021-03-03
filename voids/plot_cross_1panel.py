import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
import distinct_colours

fontsize = 22
#opts = ["shuff","partial_env_cw","partial_s2r","partial_vani","partial_tot_pot"]
opts = ["norm"]
#options = np.array(['basic HOD','environment',r'$\sigma^2 R_{\rm halfmass}$','vel. anisotropy','tot. potential'])
#options = np.array(['normal'])
options = np.array([r'${\rm SAM}$'])
offsets = [-0.4,-0.2,0.2,0.4]
colors = distinct_colours.get_distinct(len(opts))

bin_centers = np.load("data/bin_cents.npy")
cross_mean = np.load("data/cross_true_mean.npy")
cross_err = np.load("data/cross_true_error.npy")

# definitions for the axes
left, width = 0.14, 0.85#0.1, 0.65
bottom, height = 0.1, 0.25#0.2#65
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.6]
rect_histx = [left, bottom, width, height]

# start with a rectangular Figure
plt.figure(figsize=(9, 10))
ax_scatter = plt.axes(rect_scatter)
#(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)

for i in range(len(opts)):
    opt_name = options[i]
    color = colors[i]
    opt = opts[i]
    cross_opt_mean = np.load("data/cross_"+opt+"_mean.npy")
    cross_opt_err = np.load("data/cross_"+opt+"_error.npy")

    rat_opt_mean = np.load("data/rat_cross_"+opt+"_mean.npy")
    rat_opt_err = np.load("data/rat_cross_"+opt+"_error.npy")

    #ax_histx.tick_params(direction='out')#(direction='in', labelbottom=False)

    # the scatter plot:
    power = 0

    if i == 0:
        #ax_scatter.errorbar(bin_centers,cross_mean*bin_centers**power,yerr=cross_err*bin_centers**power,color='k',ls='-',linewidth=2.,fmt='o',capsize=4,label="TNG300")
        ax_scatter.plot(bin_centers,cross_mean*bin_centers**power,linewidth=1.,color='black',label="${\\rm TNG}$")
        ax_scatter.fill_between(bin_centers,(cross_mean+cross_err)*bin_centers**power,(cross_mean-cross_err)*bin_centers**power,alpha=0.1, edgecolor='black', facecolor='black')
        
        print(cross_opt_mean/cross_mean)
        
        ax_scatter.plot(bin_centers,cross_opt_mean*bin_centers**power,linewidth=1.,color='#1B2ACC',label=opt_name)
        ax_scatter.fill_between(bin_centers,(cross_opt_mean+cross_opt_err)*bin_centers**power,(cross_opt_mean-cross_opt_err)*bin_centers**power,alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')
    else:
        offset = 0#offset = offsets[i-1]
        #ax_scatter.errorbar(bin_centers+offset,cross_opt_mean*bin_centers**power,yerr=cross_opt_err*bin_centers**power,color=color,ls='-',linewidth=2.,fmt='o',capsize=4,label=opt_name)
        ax_scatter.errorbar(bin_centers+offset,np.ones(len(bin_centers)),yerr=np.ones(len(bin_centers))/20.,color=color,ls='-',linewidth=2.,fmt='o',capsize=4,label=opt_name)
    ax_scatter.set_ylabel(r"$\xi^{gv}_{\mathrm{model}}(r)$",fontsize=fontsize+6)
    ax_scatter.set_xscale('log')
    #ax_scatter.set_yscale('log')

    # now determine nice limits by hand:
    ax_scatter.set_xlim([0.8,20])
    ax_scatter.set_ylim([-1.2,0.25])
    ax_scatter.legend(fontsize=fontsize)

    line = np.linspace(0,50,3)
    ax_histx.plot(line,np.ones(len(line)),'k--')

    if i == 0:
        ax_histx.plot(bin_centers,rat_opt_mean,linewidth=1.,color='#1B2ACC',label=opt_name)
        ax_histx.fill_between(bin_centers,(rat_opt_mean+rat_opt_err),(rat_opt_mean-rat_opt_err),alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')
    else:
        offset = offsets[i-1]
        ax_histx.errorbar(bin_centers+offset,rat_opt_mean,yerr=rat_opt_err,color=color,ls='-',linewidth=2.5,fmt='o',capsize=4)

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histx.set_xscale('log')
    ax_histx.set_xlabel(r"$r \ [\mathrm{Mpc}/h]$",fontsize=fontsize+6)
    #ax_histx.set_ylabel(r'${\rm Ratio}$',fontsize=fontsize+6)
    ax_histx.set_ylabel(r'$\xi^{gv}_{\mathrm{SAM}}/\xi^{gv}_{\mathrm{TNG}}$',fontsize=fontsize+6)
    ax_histx.set_ylim([0.65,1.35])

plt.savefig("all_cross_voids.png")
plt.show()
