import numpy as np
import matplotlib.pyplot as plt

# matplotlib settings
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times'],'size':18})
rc('text', usetex=True)

import plotparams
plotparams.buba()

#halo_type = 'fof'
halo_type = 'tophat'

gas_mass = 0.000743736 # 10^10 Msun/h
dm_mass = 0.00398342749867548 # 10^10 Msun/h

n_snaps = 50
snap_start = 50
snapshots = np.arange(snap_start, snap_start+n_snaps)[::-1]

dm_nr_high = np.load("data/dm_nr_high_"+halo_type+".npy")
dm_nr_low = np.load("data/dm_nr_low_"+halo_type+".npy")
gas_nr_high = np.load("data/gas_nr_high_"+halo_type+".npy")
gas_nr_low = np.load("data/gas_nr_low_"+halo_type+".npy")
assert dm_nr_high.shape[1] == n_snaps


dm_nr_low[dm_nr_low == 0] = 1.
dm_nr_high[dm_nr_high == 0] = 1.
gas_content_low = np.mean(gas_nr_low/dm_nr_low, axis=0)*(gas_mass/dm_mass)
gas_content_high = np.mean(gas_nr_high/dm_nr_high, axis=0)*(gas_mass/dm_mass)

'''
gas_content_low = np.mean(gas_nr_low, axis=0)/np.mean(dm_nr_low, axis=0)*(gas_mass/dm_mass)
gas_content_high = np.mean(gas_nr_high, axis=0)/np.mean(dm_nr_high, axis=0)*(gas_mass/dm_mass)
'''
assert len(gas_content_low) == n_snaps

plt.figure(figsize=(9, 7))

plt.scatter(snapshots, gas_content_low, color='#CC6677', marker='*', s=40, label=r'Low env. at $z = 0$')
plt.scatter(snapshots, gas_content_high, color='dodgerblue', marker='*', s=40, label=r'High env. at $z = 0$')

plt.legend(frameon=False)
plt.xlabel(r'${\rm Snapshot \ number}$')
plt.ylabel(r'$\langle M_{\rm gas}$/$M_{\rm dm} \rangle$')
plt.savefig("figs/gas_content_"+halo_type+".png")
plt.show()

