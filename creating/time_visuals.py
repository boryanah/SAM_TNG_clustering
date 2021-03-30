import os
import glob

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as ml

from sphviewer.tools import camera_tools
from sphviewer.tools import Blend
import sphviewer as sph


def get_normalized_image(image, vmin=None, vmax=None):
    if(vmin == None):
        vmin = np.min(image)/4
    if(vmax == None):
        vmax = np.max(image)/4
    #print("min, max image = ", np.min(image), np.max(image))
    image = np.clip(image, vmin, vmax)
    image = (image-vmin)/(vmax-vmin)

    return image

# location where your particles are stored
visual_dir = "/mnt/gosling1/boryanah/TNG300/visuals/"

# snapshot information
snap_num = 50
snap_start = (100-snap_num)
snap_mid = snap_start+snap_num//2

# type of environment
type_env = 'high'
#type_env = 'low'

# saturation parameters
'''
vmin_gas = 0.
vmax_gas = 10000.
vmin_dm = 0.
vmax_dm = 10000.
'''
vmin_gas = None
vmax_gas = None
vmin_dm = None
vmax_dm = None

# select halo index
inds_type = np.load(f"visuals/inds_{type_env:s}.npy")


for halo_id in inds_type:
    #halo_id = inds_type[0]
    print("halo index = ", halo_id)
    
    # load all filenames associated with this halo
    fns_dm = sorted(glob.glob(visual_dir+f"dm_{type_env:s}_{halo_id:d}_pos_*.npy"))
    fns_gas = sorted(glob.glob(visual_dir+f"gas_{type_env:s}_{halo_id:d}_pos_*.npy"))

    assert len(fns_dm) == snap_num, "Different expectation for number of snapshots"
    assert len(fns_gas) == snap_num, "Different expectation for number of snapshots"

    # record the center of mass location at 3 distinct times
    cms = []
    pos_dm = np.load(visual_dir+f"dm_{type_env:s}_{halo_id:d}_pos_{snap_start:d}.npy")/1000.
    cm_dm = np.mean(pos_dm, axis=0)
    cms.append(cm_dm)
    pos_dm = np.load(visual_dir+f"dm_{type_env:s}_{halo_id:d}_pos_{snap_mid:d}.npy")/1000.
    cm_dm = np.mean(pos_dm, axis=0)
    cms.append(cm_dm)
    pos_dm = np.load(visual_dir+f"dm_{type_env:s}_{halo_id:d}_pos_99.npy")/1000.
    cm_dm = np.mean(pos_dm, axis=0)
    cms.append(cm_dm)
    #pos_stars = np.load(visual_dir+f"stars_{type_env:s}_{halo_id:d}_pos_99.npy")/1000.

    # where are you pointing the camera towards
    targets = cms

    # parameters
    r = 2.5
    id_frames = np.array([0, 2, 5, 7, 10, 12, 15, 17, 20])*10/4

    # define the camera parameters
    anchors = {}
    anchors['sim_times'] =  np.arange(len(id_frames)) # how long does each last?
    anchors['id_frames'] =  id_frames # spread over how many frames
    anchors['r']         =  [r, 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same'] # distance
    anchors['id_targets']=  [0, 'pass', 'pass', 'pass', 1, 'pass', 'pass', 'pass', 2] # where are you pointing towards
    anchors['t']         =  [0, 'same', 'same', 'same', 'same', 'same', 'same', 'same', 0] # theta
    anchors['p']         =  [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 0] # phi
    anchors['zoom']      =  [1., 'same', 'same', 'same', 1, 'same', 'same', 'same', 1] # zooming in and out
    anchors['extent']    =  [5., 'same', 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    data = camera_tools.get_camera_trajectory(targets, anchors)

    # to make it faster
    #data = data[:20]

    # snap counter
    snap = 0
    new_snap = np.max(id_frames)//snap_num

    # go through each frame
    for i, datum in enumerate(data):
        print("i = %d"%i, end = '\r')
        
        # number of pixels
        datum['xsize'] = 500
        datum['ysize'] = 500
        datum['roll'] = 1

        if i % new_snap == 0:
            print("new snapshot, new file = ", snap_start+snap, fns_dm[snap], end = '\r')
            # load the particles in one of the snapshots
            pos_dm = np.load(fns_dm[snap])/1000.
            pos_gas = np.load(fns_gas[snap])/1000.

            # load the particles into sph
            P_dm = sph.Particles(pos_dm, np.ones(pos_dm.shape[0]))
            P_gas = sph.Particles(pos_gas, np.ones(pos_gas.shape[0]))

            # load the particles into the Scene
            S_dm = sph.Scene(P_dm)
            S_gas = sph.Scene(P_gas)

            snap += 1

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


        if i == 0:
            # trying to set the saturation levels
            vmin_dm = np.min(img_dm)/2
            vmax_dm = np.max(img_dm)/2
            vmin_gas = np.min(img_gas)/2
            vmax_gas = np.max(img_gas)/2
        
        # apply color map
        rgb_dm  = plt.cm.Greys_r(get_normalized_image(img_dm, vmin_dm, vmax_dm))
        rgb_gas = plt.cm.magma(get_normalized_image(img_gas, vmin_gas, vmax_gas))
        
        # overlay the different species
        blend = Blend.Blend(rgb_dm, rgb_gas)
        #output = blend.Overlay()
        output = blend.Screen()

        # save output
        plt.imsave(f'img/time_dm_gas_{type_env:s}_{halo_id:d}_{i:04d}.png', output)
