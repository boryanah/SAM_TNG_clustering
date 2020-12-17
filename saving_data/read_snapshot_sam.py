import site
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import illustris_sam as ilsam
import os

# path where info kept
basePath = '/mnt/alan1/boryanah/SAM_subvolumes/'

# halo and subhalo fields
fields_halo = ['HalopropC_nfw', 'HalopropMvir', 'HalopropSpin', 'HalopropRedshift']#, 'HalopropIndex_Snapshot']
fields_gal = ['GalpropMvir', 'GalpropMstar', 'GalpropPos', 'GalpropRhalo', 'GalpropSatType', 'GalpropSfr', 'GalpropSfrave100myr', 'GalpropSigmaBulge', 'GalpropVel', 'GalpropHaloIndex_Snapshot', 'GalpropRedshift']

# fields you need to multiply by a factor
h_fields = ['GalpropMvir','GalpropMstar','HalopropMvir','GalpropPos','GalpropRhalo']
m_fields = ['GalpropMvir','GalpropMstar','HalopropMvir']

# multiplication factors
h = 0.6774
m_factor = 1.e9
snap = 99
snap_str = '_%d'%snap if snap != 99 else ''

# dictionary with names used to save
halo_dic = {}
subhalo_dic = {}

for field in fields_halo:
    halo_dic[field] = field+snap_str
for field in fields_gal:
    subhalo_dic[field] = field+snap_str

# exceptions
subhalo_dic['GalpropMvir'] = 'GalpropMhalo'+snap_str
subhalo_dic['GalpropHaloIndex_Snapshot'] = 'GalpropHaloIndex'+snap_str
    
# subvolumes for extracting info
subvolume_list = []
for i in range(5):
    for j in range(5):
        for k in range(5):
            subvolume_list.append([i, j, k])

# load snapshots
SAM_subhalos = ilsam.groupcat.load_snapshot_subhalos(basePath, snap, subvolume_list,fields=fields_gal,matches=True)
SAM_halos = ilsam.groupcat.load_snapshot_halos(basePath, snap, subvolume_list,fields=fields_halo,matches=True)

# BIG EXCEPTION -- keeping old convention
field = 'HalopropSubfindID'
np.save(basePath+field+snap_str+".npy",SAM_subhalos['GalpropSubfindIndex_DM'][SAM_subhalos['GalpropSatType'] == 0])
quit()

# save rest of the fields
for field in fields_halo:
    array = SAM_halos[field]
    if field in h_fields:
        array *= h
    if field in m_fields:
        array *= m_factor
    np.save(basePath+halo_dic[field]+".npy",array)


for field in fields_gal:
    array = SAM_subhalos[field]
    if field in h_fields:
        array *= h
    if field in m_fields:
        array *= m_factor
    np.save(basePath+subhalo_dic[field]+".npy",array)

# bijective matches
field = 'HalopropFoFIndex_FP'
np.save(basePath+field+snap_str+".npy",SAM_halos['HalopropFoFIndex_DM'])
field = 'HalopropFoFIndex_DM'
np.save(basePath+field+snap_str+".npy",SAM_halos['HalopropFoFIndex_FP'])
field = 'GalpropSubfindIndex_FP'
np.save(basePath+field+snap_str+".npy",SAM_subhalos['GalpropSubfindIndex_DM'])
field = 'GalpropSubfindIndex_DM'
np.save(basePath+field+snap_str+".npy",SAM_subhalos['GalpropSubfindIndex_FP'])


os.remove(basePath+"HalopropRedshift"+snap_str+".npy")
os.remove(basePath+"GalpropRedshift"+snap_str+".npy")
quit()
header = ilsam.groupcat.load_header(basePath, [[0, 0, 0],[0, 0, 1]])
print(header.keys())
