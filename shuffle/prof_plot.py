import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

color_sam = 'dodgerblue'
color_hydro = '#CC6677'

rel_pos_sam = np.load("data/rel_pos_gals_norm_sam.npy")
rel_pos_fp = np.load("data/rel_pos_gals_norm_fp.npy")

d_sam = np.sqrt(np.sum(rel_pos_sam**2, axis=1))
d_fp = np.sqrt(np.sum(rel_pos_fp**2, axis=1))
d_sam = d_sam[d_sam > 0.]
d_fp = d_fp[d_fp > 0.]

# can we make their numbers equal
np.random.shuffle(d_fp)
d_fp = d_fp[d_fp < 25]
d_fp = d_fp[:len(d_sam)]

bins = np.logspace(-1.5, 2., 21)
vol_bins = 4/3.*np.pi*(bins[1:]**3-bins[:-1]**3)
bin_cents = 0.5*(bins[1:]+bins[:-1])
hist_sam, bin_edges = np.histogram(d_sam, bins)
hist_fp, bin_edges = np.histogram(d_fp, bins)
print("sum sam = ", hist_sam.sum())
print("sum fp = ", hist_fp.sum())
hist_sam = (hist_sam).astype(float)
hist_fp = (hist_fp).astype(float)
hist_sam /= vol_bins
hist_fp /= vol_bins

plt.figure(figsize=(9,7))
plt.plot(bin_cents, hist_sam, color=color_sam, label=r"${\rm SAM}$")
plt.plot(bin_cents, hist_fp, color=color_hydro, label=r"${\rm TNG}$")
plt.xlabel(r"$r/R_{\rm vir}$")
plt.ylabel(r"$n(r)$")
plt.xscale('log')
plt.yscale('log')
plt.xlim([0.01, 100])
#plt.ylim([1.e-4, 1e3])
plt.legend()
plt.savefig("figs/prof.png")
plt.show()
