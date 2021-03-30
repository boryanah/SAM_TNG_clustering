import os

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as ml

from sphviewer.tools import camera_tools
from sphviewer.tools import Blend
import sphviewer as sph


def get_normalized_image(image, vmin=None, vmax=None):
    if(vmin == None):
        vmin = np.min(image)
    if(vmax == None):
        vmax = np.max(image)
    #print("min, max image = ", np.min(image), np.max(image))
    image = np.clip(image, vmin, vmax)
    image = (image-vmin)/(vmax-vmin)

    return image

# Load your own set of data (particles)
visual_dir = "/mnt/gosling1/boryanah/TNG300/visuals/"
pos_dm = np.load(visual_dir+"dm_high_178_pos_99.npy")/1000.
pos_gas = np.load(visual_dir+"gas_high_178_pos_99.npy")/1000.
#pos_stars = np.load(visual_dir+"stars_high_178_pos_99.npy")/1000.

# saturation parameters
vmin_gas = 0.
vmax_gas = 10000.
vmin_dm = 0.
vmax_dm = 10000.
'''
vmin_gas = None
vmax_gas = None
vmin_dm = None
vmax_dm = None
'''

# create array with masses
n_parts_dm = pos_dm.shape[0]
mass_dm = np.ones(n_parts_dm)
n_parts_gas = pos_gas.shape[0]
mass_gas = np.ones(n_parts_gas)
print("num parts dm = ", n_parts_dm)
print("num parts gas = ", n_parts_gas)

# load particles
P_dm = sph.Particles(pos_dm, mass_dm)
cm_dm = np.mean(pos_dm, axis=0)
P_gas = sph.Particles(pos_gas, mass_gas)
cm_gas = np.mean(pos_gas, axis=0)

# where are you pointing the camera towards
targets = [cm_dm]

# parameters
r = 3.

# define the camera parameters
anchors = {}
anchors['sim_times'] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # how long does each last?
anchors['id_frames'] =  np.array([0, 2, 5, 7, 10, 12, 15, 17, 20])*10 # spread over how many frames
anchors['r']         =  [r, 'same', r, 'same', r, 'same', r, 'same', 'same'] # distance
anchors['id_targets']=  [0, 'same', 0, 'same', 0, 'same', 0, 'same', 'same'] # where are you pointing towards
anchors['t']         = [0, 'same', 'same', 'same', 'same', 'same', 'same', 'same', 0] # theta
anchors['p']         = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 360] # phi
anchors['zoom']      = [1., 'same', 1., 'same', 1, 'same', 1., 'same', 'same'] # zooming in and out
anchors['extent']    = [5., 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same']
data = camera_tools.get_camera_trajectory(targets, anchors)

# load the particles in the Scene
S_dm = sph.Scene(P_dm)
S_gas = sph.Scene(P_gas)

# to make it faster
#data = data[:5]

# go through each frame
for i, datum in enumerate(data):
    print("i = %d"%i, end = '\r')

    # number of pixels
    datum['xsize'] = 500
    datum['ysize'] = 500
    datum['roll'] = 1
    
    # update the camera
    S_dm.update_camera(**datum)
    S_gas.update_camera(**datum)
    
    # render the image
    R_dm = sph.Render(S_dm)
    R_gas = sph.Render(S_gas)

    # extract image
    img_dm = R_dm.get_image()
    img_gas = R_gas.get_image()
    #R.set_logscale()

    # apply color map
    rgb_dm  = plt.cm.Greys_r(get_normalized_image(img_dm, vmin_dm, vmax_dm))
    rgb_gas = plt.cm.magma(get_normalized_image(img_gas, vmin_gas, vmax_gas))

    # overlay the different species
    blend = Blend.Blend(rgb_dm, rgb_gas)
    #output = blend.Overlay()
    output = blend.Screen()
    
    # save output
    plt.imsave('img/dm_gas_%04d.png'%i, output)
