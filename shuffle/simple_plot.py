import matplotlib.pyplot as plt
import numpy as np
import plotparams
plotparams.buba()

color_sam = 'dodgerblue'
color_hydro = '#CC6677'

bin_cents = np.load("data/bin_centers.npy")
hydro_mean = np.load("data/mean_hydro.npy")
sam_mean = np.load("data/mean_sam.npy")
hydro_err = np.load("data/err_hydro.npy")
sam_err = np.load("data/err_sam.npy")

plt.figure(figsize=(7,6))
plt.errorbar(bin_cents, hydro_mean*bin_cents**2, yerr=hydro_err*bin_cents**2, color=color_hydro, ls='-', fmt='o', capsize=4, label=r'${\rm TNG}$')
plt.errorbar(bin_cents, sam_mean*bin_cents**2, yerr=sam_err*bin_cents**2, color=color_sam, ls='-', fmt='o', capsize=4, label=r'${\rm SAM}$')
plt.legend()
plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.ylabel(r"$\xi(r) r^2$")
plt.xscale('log')
plt.savefig("figs/corrfunc.png")
plt.show()
