import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import Corrfunc

import matplotlib.pyplot as plt
import matplotlib.ticker

import plotparams
#plotparams.default()
plotparams.buba()

proxy = 'm200m'
opts = np.array(['partial_env_cw','partial_vani','partial_s2r','partial_tot_pot'])
lab_opts = np.array(['environment','vel. anisotropy',r'$\sigma^2 R_{\rm halfmass}$', 'tot. potential'])

N = len(opts)

rp_mids = np.load("../data_jack/rp_mids.npy")
rat_shuff_mean = np.load("../data_jack/ds_rat_"+proxy+"_shuff_mean.npy")
rat_shuff_err = np.load("../data_jack/ds_rat_"+proxy+"_shuff_err.npy")

nrows = 1
ncols = N//nrows
plt.subplots(nrows,ncols,figsize=(ncols*4.7,nrows*5.7))
for i in range(N):
    opt = opts[i]
    lab_name = lab_opts[i]#'-'.join(opt.split('_'))
    plt.subplot(nrows,ncols,i+1)
    rat_mean = np.load("../data_jack/ds_rat_"+proxy+"_"+opt+"_mean.npy")
    rat_err = np.load("../data_jack/ds_rat_"+proxy+"_"+opt+"_err.npy")
    
    plt.plot(rp_mids,np.ones(len(rp_mids)),'k--',alpha=0.2)

    if i == 0:
        fid_name = 'basic HOD'
    else:
        fid_name = ''
        
    plt.plot(rp_mids,rat_shuff_mean,linewidth=1.,color='#1B2ACC',label=fid_name)
    plt.fill_between(rp_mids,rat_shuff_mean+rat_shuff_err,rat_shuff_mean-rat_shuff_err,alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')

    plt.errorbar(rp_mids,rat_mean,yerr=rat_err,ls='-',c='#CC6677',fmt='o',capsize=4,label=lab_name)

    plt.legend(loc='upper left',ncol=1)
    plt.xscale('log')
    if i == 0 or i==N//nrows:
        plt.ylabel(r'$\xi(r)_{\rm model} / \xi(r)_{\rm TNG300}$') # Daniel
    else:
        plt.gca().axes.yaxis.set_ticklabels([])
    # for getting rid of the numbers on x axis
    #if i >= N-ncols:
    #    plt.gca().axes.xaxis.set_ticklabels([])
    plt.xlabel(r'$r$ [Mpc/h]')

    tick_spacing = 0.05#0.2
    plt.gca().axes.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.1,.2,.3,0.4,.5,0.6,.7,0.8,.9),numticks=10)
    plt.gca().axes.xaxis.set_minor_locator(locmin)
    plt.gca().axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.gca().axes.xaxis.set_major_locator(ticker.LogLocator(base=10))
    plt.gca().axes.xaxis.set_major_locator(ticker.FixedLocator([0.1,1,10]))
    plt.ylim([0.9,1.15])
    plt.xlim([0.27,25])

plt.savefig('jack_m200m_delta_sigma.pdf')
plt.show()
plt.close()
