import numpy as np
import matplotlib.pyplot as plt
import plotparams
#plotparams.default()
plotparams.buba()

#opts = np.array(['shuff', 'env_mass', 'min_pot', 'partial_vani', 'partial_env_cw', 'partial_tot_pot', 'conc', 'spin'])
#opts = np.array(['shuff', 'env_mass', 'min_pot', 'partial_vani', 'partial_env_cw', 'partial_tot_pot', 'partial_s2r', 'partial_fenv'])
opts = np.array(['shuff','partial_env_cw','partial_vani','partial_s2r', 'partial_tot_pot'])
lab_opts = np.array(['basic HOD','environment','vel. anisotropy',r'$\sigma^2 R_{\rm halfmass}$', 'tot. potential'])
types = ['bias','corr. coeff.']

bin_centers = np.load("../data_gm/bin_fid.npy")
bias_fid = np.load("../data_gm/bias_mean_fid.npy")
corr_coeff_fid = np.load("../data_gm/corr_coeff_mean_fid.npy")
#bias_fid = np.load("../data_gm/corr_gg_mean_fid.npy")*bin_centers**2
#bias_error_fid = np.load("../data_gm/corr_gg_error_fid.npy")*bin_centers**2

bias_error_fid = np.load("../data_gm/bias_error_fid.npy")
corr_coeff_error_fid = np.load("../data_gm/corr_coeff_error_fid.npy")

nprops = len(opts)
nrows = 2
ncols = nprops
ntot = nrows*ncols
plt.subplots(nrows,ncols,figsize=(ncols*4.8,nrows*5))
plot_no = 0
for i in range(nprops):
    for i_type in range(2):
        opt = opts[i]
        lab_opt = lab_opts[i]#"-".join(opt.split('_'))+" "+types[i_type]
        
        print(opt)

            
        if i_type == 0:
            bias = np.load("../data_gm/bias_mean_"+opt+".npy")
            bias_error = np.load("../data_gm/bias_error_"+opt+".npy")
        else:    
            bias = np.load("../data_gm/corr_coeff_mean_"+opt+".npy")
            bias_error = np.load("../data_gm/corr_coeff_error_"+opt+".npy")
        
        plot_no = i_type*ncols+i+1

        plt.subplot(nrows,ncols,plot_no)
        plt.plot(bin_centers,np.ones(len(bias)),'k--',linewidth=2.)
        

        if i == 0:
            # blue ones
            shuff_color2 = '#089FFF'
            shuff_color = '#1B2ACC'
            lab_shuff = lab_opts[0]
            if i_type == 0:
                bias_shuff = bias.copy()
                bias_error_shuff = bias_error.copy()
                plt.plot(bin_centers,bias_shuff,linewidth=2.,color=shuff_color,label=lab_shuff)
                plt.fill_between(bin_centers,bias_shuff+bias_error_shuff,bias_shuff-bias_error_shuff,alpha=0.1, edgecolor=shuff_color, facecolor=shuff_color2)
            else:
                corr_coeff_shuff = bias.copy()
                corr_coeff_error_shuff = bias_error.copy()
                plt.plot(bin_centers,corr_coeff_shuff,linewidth=2.,color=shuff_color,label=lab_shuff)
                plt.fill_between(bin_centers,corr_coeff_shuff+corr_coeff_error_shuff,corr_coeff_shuff-corr_coeff_error_shuff,alpha=0.1, edgecolor=shuff_color, facecolor=shuff_color2)
        else:
            # orange ones
            shuff_color = '#CC6677'
            plt.errorbar(bin_centers,bias,yerr=bias_error,ls='-',c=shuff_color,fmt='o',capsize=4,label=lab_opt)

        
        # always plot the fiducial in black
        if i_type == 0:
            fid_color = 'k'#yell='#DDCC77'#'gray'
            fid_color2 = 'k'#yell='#DDCC77'#'gray'
            if i == 0:
                lab_fid = 'TNG300'
            else:
                lab_fid = ''
            plt.plot(bin_centers,bias_fid,linewidth=2.,color=fid_color,label=lab_fid)
            plt.fill_between(bin_centers,bias_fid+bias_error_fid,bias_fid-bias_error_fid,alpha=0.1, edgecolor=fid_color, facecolor=fid_color2)
            plt.legend(loc='upper left',frameon=False)
        else:
            lab_fid = ''
            plt.plot(bin_centers,corr_coeff_fid,linewidth=2.,color=fid_color,label=lab_fid)
            plt.fill_between(bin_centers,corr_coeff_fid+corr_coeff_error_fid,corr_coeff_fid-corr_coeff_error_fid,alpha=0.1, edgecolor=fid_color, facecolor=fid_color2)
            
        if i_type == 0:
            plt.ylim([0.95,1.8])#plt.ylim([0.7,1.5])
        if i_type == 1:
            plt.ylim([0.92,1.12])#plt.ylim([0.5,1.32])
        #origplt.xlim([.7,15])
        plt.xlim([.7,15])
        plt.xscale('log')

        if plot_no >= ntot-ncols+1:
            plt.xlabel(r'$r$ [Mpc/h]')
        if plot_no%ncols == 1 and i_type == 0:
            plt.ylabel(r'$\tilde b (r) = (\xi_{\rm gg}/\xi_{\rm mm})^{1/2}$')
        elif plot_no%ncols == 1 and i_type == 1:
            plt.ylabel(r'$\tilde r (r) = \xi_{\rm gm}/(\xi_{\rm gg} \xi_{\rm mm})^{1/2}$')
        else:
            plt.gca().axes.yaxis.set_ticklabels([])

plt.savefig("bias_corr_coeff.pdf")
plt.show()
