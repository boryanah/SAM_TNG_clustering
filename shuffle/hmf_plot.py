import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

color_sam = 'dodgerblue'
color_hydro = '#CC6677'
bin_cents = np.load("data/bin_hmf.npy")
hmf_sam = np.load("data/hmf_sam.npy")
#hmf_fp = np.load("data/hmf_fp.npy")
hmf_fp = np.load("data/hmf_dm.npy")


plt.subplots(2, 1, figsize=(9, 12))
plt.subplot(2, 1, 1)
plt.loglog(bin_cents, hmf_sam, color=color_sam, label="${\\rm SAM}$")
plt.loglog(bin_cents, hmf_fp, color=color_hydro, label="${\\rm TNG \ DMO}$")
plt.legend()
plt.ylabel("$N_{\\rm halo}$")
plt.subplot(2, 1, 2)
plt.plot(bin_cents, np.ones(len(bin_cents)), 'k--')
plt.plot(bin_cents, hmf_fp/hmf_sam)
plt.ylabel("$N_{\\rm halo, TNGO}/N_{\\rm halo, SAM}$")
plt.xscale('log')
plt.xlim(2.e10, 2.e15)
plt.savefig("figs/hmf.png")
plt.show()
