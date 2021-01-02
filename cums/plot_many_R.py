import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

proxy = 'm200m'
opts = np.array(['partial_env_cw','partial_vani','partial_s2r','partial_tot_pot'])
lab_opts = np.array(['environment','vel. anisotropy',r'$\sigma^2 R_{\rm halfmass}$', 'tot. potential'])
types = ['second moment','third moment']

old = ''#'old_many/'#''#

bin_centers = np.load("data_many/"+old+"rs.npy")

# if not ratios
#second_fid = np.load("data_many/"+old+"second_true_mean.npy")
#second_error_fid = np.load("data_many/"+old+"second_true_err.npy")
third_fid = np.load("data_many/"+old+"third_true_mean.npy")
third_error_fid = np.load("data_many/"+old+"third_true_err.npy")


second_fid = np.load("data_many/"+old+"second_rat_"+proxy+"_shuff_mean.npy")
second_error_fid = np.load("data_many/"+old+"second_rat_"+proxy+"_shuff_err.npy")
third_fid = np.load("data_many/"+old+"third_rat_"+proxy+"_shuff_mean.npy")
third_error_fid = np.load("data_many/"+old+"third_rat_"+proxy+"_shuff_err.npy")

nprops = len(opts)
nrows = 2
ncols = nprops
ntot = nrows*ncols
plt.subplots(nrows,ncols,figsize=(ncols*4.8,nrows*5))
plot_no = 0
for i in range(nprops):
    for i_type in range(2):
        lab_opt = lab_opts[i]#"-".join(opt.split('_'))+" "+types[i_type]
        opt = opts[i]
        print(opt)
        
        if i_type == 0:
            #second = np.load("data_many/"+old+"second_"+proxy+"_"+opt+"_mean.npy")
            second = np.load("data_many/"+old+"second_rat_"+proxy+"_"+opt+"_mean.npy")
            second_err = np.load("data_many/"+old+"second_rat_"+proxy+"_"+opt+"_err.npy")
        else:
            second = np.load("data_many/"+old+"third_rat_"+proxy+"_"+opt+"_mean.npy")
            second_err = np.load("data_many/"+old+"third_rat_"+proxy+"_"+opt+"_err.npy")
        
        plot_no = i_type*ncols+i+1

        plt.subplot(nrows,ncols,plot_no)
        plt.plot(np.linspace(0,10,len(second)),np.ones(len(second)),'k--',linewidth=2.)
        
        # orange ones
        plt.errorbar(bin_centers,second,yerr=second_err,ls='-',c='#CC6677',fmt='o',capsize=4,label=lab_opt)

        if plot_no == 1:
            lab_fid = 'basic HOD'
        else:
            lab_fid = ''
        
        # always plot the fiducial
        if i_type == 0:
            plt.plot(bin_centers,second_fid,linewidth=2.,color='#1B2ACC',label=lab_fid)
            plt.fill_between(bin_centers,second_fid+second_error_fid,second_fid-second_error_fid,alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')
        else:
            plt.plot(bin_centers,third_fid,linewidth=2.,color='#1B2ACC',label=lab_fid)
            plt.fill_between(bin_centers,third_fid+third_error_fid,third_fid-third_error_fid,alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')
            
        plt.legend(loc='upper left',fontsize=18)
        if i_type == 0:
            plt.ylim([0.9,1.1])
        if i_type == 1:
            plt.ylim([0.78,1.31])
        #origplt.xlim([.7,15])
        plt.xlim([2.5,8.5])
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xticks(np.arange(int(min(bin_centers)), int(max(bin_centers))+1, 1.0))
        
        if plot_no >= ntot-ncols+1:
            plt.xlabel(r'$R$ [Mpc/h]')
        if plot_no%ncols == 1 and i_type == 0:
            plt.ylabel(r'$\langle \delta_{R,\ \textrm{model}}^2 \rangle/\langle \delta_{R,\ \textrm{TNG300}}^2 \rangle$',fontsize=22)
        elif plot_no%ncols == 1 and i_type == 1:
            plt.ylabel(r'$\langle \delta_{R,\ \textrm{model}}^3 \rangle/\langle \delta_{R,\ \textrm{TNG300}}^3 \rangle$',fontsize=22)
        else:
            plt.gca().axes.yaxis.set_ticklabels([])

plt.savefig("second_third.pdf")
plt.show()
