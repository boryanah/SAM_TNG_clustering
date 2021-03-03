import matplotlib.pyplot as plt
import numpy as np
import sys
import plotparams
plotparams.buba()

bin_centers = np.load("data/bin_centers.npy")

hist_g_opt_mean = np.load("data/void_true_mean.npy")
hist_g_opt_err  = np.load("data/void_true_err.npy")

hist_g_mean = np.load("data/void_norm_mean.npy")
hist_g_err  = np.load("data/void_norm_err.npy")

fig = plt.figure(figsize=(6.4,5.5))

plt.plot(bin_centers,hist_g_mean,linewidth=1.,color='black',label="TNG300")
plt.fill_between(bin_centers,hist_g_mean+hist_g_err,hist_g_mean-hist_g_err,alpha=0.1, edgecolor='black', facecolor='black')#edgecolor='#1B2ACC', facecolor='#089FFF')
plt.plot(bin_centers,hist_g_opt_mean,linewidth=1.,color='dodgerblue',label="SAM")
plt.legend()
plt.yscale('log')
plt.xlim([10,22.6]) # og
plt.ylim([0.1,800])
#plt.xticks(np.arange(int(min(bin_centers)), int(max(bin_centers))+1, 2.0))# og
#plt.xlim([5,16])
plt.xlabel("Size of void [Mpc/h]")
plt.ylabel("Number of Voids")
plt.savefig("true_voids.pdf")
plt.show()

