import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

proxy = 'm200m'
#opts = np.array(['partial_env_cw','partial_vani','partial_s2r','partial_tot_pot'])
opts = np.array(['norm'])
#lab_opts = np.array(['environment','vel. anisotropy',r'$\sigma^2 R_{\rm halfmass}$', 'tot. potential'])
lab_opts = np.array(['normal'])
types = ['second moment','third moment']

bin_centers = np.load("data/rs.npy")

# if not ratios
#second_fid = np.load("data/second_true_mean.npy")
#second_error_fid = np.load("data/second_true_err.npy")
third_fid = np.load("data/third_true_mean.npy")
third_error_fid = np.load("data/third_true_err.npy")


second_fid = np.load("data/second_rat_norm_mean.npy")
second_error_fid = np.load("data/second_rat_norm_err.npy")
third_fid = np.load("data/third_rat_norm_mean.npy")
third_error_fid = np.load("data/third_rat_norm_err.npy")

nprops = len(opts)
nrows = 2
ncols = nprops
ntot = nrows*ncols
plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*4.5))
plot_no = 0
for i in range(nprops):
    for i_type in range(2):
        lab_opt = lab_opts[i]
        opt = opts[i]
        print(opt)
        
        if i_type == 0:
            second = np.load("data/second_rat_"+opt+"_mean.npy")
            second_err = np.load("data/second_rat_"+opt+"_err.npy")
        else:
            second = np.load("data/third_rat_"+opt+"_mean.npy")
            second_err = np.load("data/third_rat_"+opt+"_err.npy")
        
        plot_no = i_type*ncols+i+1

        plt.subplot(nrows,ncols,plot_no)
        plt.plot(np.linspace(0,20,len(second)),np.ones(len(second)),'k--',linewidth=2.)
        
        # orange ones
        plt.errorbar(bin_centers,second,yerr=second_err,ls='-',c='#CC6677',fmt='o',capsize=4,label=lab_opt)

        if plot_no == 1:
            lab_fid = 'basic HOD'
        else:
            lab_fid = ''
        
        # always plot the fiducial
        '''
        if i_type == 0:
            plt.plot(bin_centers,second_fid,linewidth=2.,color='#1B2ACC',label=lab_fid)
            plt.fill_between(bin_centers,second_fid+second_error_fid,second_fid-second_error_fid,alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')
        else:
            plt.plot(bin_centers,third_fid,linewidth=2.,color='#1B2ACC',label=lab_fid)
            plt.fill_between(bin_centers,third_fid+third_error_fid,third_fid-third_error_fid,alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')
        '''
        
        #plt.legend(loc='upper left',fontsize=18)
        if i_type == 0:
            plt.ylim([0.4,1.6])
            plt.gca().axes.xaxis.set_ticklabels([])
        if i_type == 1:
            plt.ylim([0.4,1.6])
        plt.xlim([3.,11.])
        plt.xticks(np.arange(int(min(bin_centers)), int(max(bin_centers))+3, 1.0))
        
        if plot_no >= ntot-ncols+1:
            plt.xlabel(r'$R_{\rm smooth} \ [{\rm Mpc}/h]$')
        if plot_no%ncols == 0 and i_type == 0:
            plt.ylabel(r'$\langle \delta_{R,\ \textrm{SAM}}^2 \rangle/\langle \delta_{R,\ \textrm{TNG}}^2 \rangle$',fontsize=22)
        elif plot_no%ncols == 0 and i_type == 1:
            plt.ylabel(r'$\langle \delta_{R,\ \textrm{SAM}}^3 \rangle/\langle \delta_{R,\ \textrm{TNG}}^3 \rangle$',fontsize=22)
        else:
            plt.gca().axes.yaxis.set_ticklabels([])

plt.savefig("second_third.png")
plt.show()
