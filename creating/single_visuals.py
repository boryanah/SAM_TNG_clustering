from sphviewer.tools import QuickView

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

from sphviewer.tools import camera_tools

import sphviewer as sph

# Load your own set of data (particles)
visual_dir = "/mnt/gosling1/boryanah/TNG300/visuals/"
#pos = np.load(visual_dir+"dm_high_178_pos_99.npy")/1000.#*2.
#pos = np.load(visual_dir+"gas_high_178_pos_99.npy")/1000.#*2.
pos = np.load(visual_dir+"stars_high_178_pos_99.npy")/1000.#*2.
#pos = pos[::10]

# create array with masses
n_parts = pos.shape[0]
mass = np.ones(n_parts)
print("num parts = ", n_parts)

# load particles
P = sph.Particles(pos, mass)
S = sph.Scene(P)
cm = np.mean(pos, axis=0)

# where are you pointing the camera towards
cm_0 = [cm[0], cm[1], cm[2]]
cm_1 = cm_0# + [-1.4, -2.55, 1.1]
targets = [cm_0, cm_1]

# parameters
r = 3.

# define the camera parameters
anchors = {}
anchors['sim_times'] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # how long does each last?
anchors['id_frames'] =  np.array([0, 2, 5, 7, 10, 12, 15, 17, 20])*10 # spread over how many frames
anchors['r']         =  [r, 'same', r, 'same', r, 'same', r, 'same', 'same'] # distance
anchors['id_targets']=  [0, 'same', 0, 'same', 0, 'same', 'same', 'same', 'same'] # where are you pointing towards
anchors['t']         = [0, 'same', 'same', 'same', 'same', 'same', 'same', 'same', 0] # phi
anchors['p']         = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 200] # theta
anchors['zoom']      = [1., 'same', 1., 'same', 1, 'same', 1., 'same', 'same'] # zooming out
anchors['extent']    = [5., 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same']
data = camera_tools.get_camera_trajectory(targets, anchors)

# to make it faster
data = data[:50]

h = 0
for i in data:
    i['xsize'] = 2000#500
    i['ysize'] = 2000#500
    i['roll'] = 1

    S = sph.Scene(P)
    S.update_camera(**i)
    R = sph.Render(S)
    
    img = R.get_image()
    R.set_logscale()
    
    plt.imsave('img/image_'+str('%04d.png'%h), img, vmin=0, vmax=2000, cmap=plt.cm.magma)
    h += 1
