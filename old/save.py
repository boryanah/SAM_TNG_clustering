import numpy as np
import h5py
import pandas as pd

#SAM_dir = '/mnt/gosling1/boryanah/SAM/TNG100-1-SAM/'
SAM_dir = '/mnt/gosling1/boryanah/SAM/TNG100-1-SAM-Recalibrated/'

typ = 'gal'#'halo'#'gal'
df = pd.read_hdf(SAM_dir+typ+'prop/'+typ+'prop_99.h5', 'df')
#df.shape (11992571, 41)
print(df.keys())

# halo
#Index(['halo_index', 'halo_id', 'roothaloid', 'orig_halo_ID', 'redshift',
#       'm_vir', 'c_nfw', 'spin', 'm_hot', 'mstar_diffuse', 'mass_ejected',
#       'mcooldot', 'maccdot_pristine', 'maccdot_reaccrete',
#       'maccdot_metal_reaccrete', 'maccdot_metal', 'mdot_eject',
#       'mdot_metal_eject', 'maccdot_radio', 'Metal_hot', 'Metal_ejected',
#       'snap_num'],
#       dtype='object')

# gal
#Index(['halo_index', 'birthhaloid', 'roothaloid', 'redshift', 'sat_type',
#      'mhalo', 'm_strip', 'rhalo', 'mstar', 'mbulge', 'mstar_merge', 'v_disk',
#      'sigma_bulge', 'r_disk', 'r_bulge', 'mcold', 'mHI', 'mH2', 'mHII',
#      'Metal_star', 'Metal_cold', 'sfr', 'sfrave20myr', 'sfrave100myr',
#      'sfrave1gyr', 'mass_outflow_rate', 'metal_outflow_rate', 'mBH',
#      'maccdot', 'maccdot_radio', 'tmerge', 'tmajmerge', 'mu_merge', 't_sat',
#      'r_fric', 'x_position', 'y_position', 'z_position', 'vx', 'vy', 'vz'],
#      dtype='object')

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

if typ == 'halo': ext = 'halo_'
else: ext = ''

#fields = ['x_position', 'y_position', 'z_position']
fields = ['rhalo','sigma_bulge']
#fields = ['c_nfw', 'spin']

for field in fields:
    value = np.array(df[field].values)
    print(value[:10])
    value = value.astype(np.float32)
    np.save(SAM_dir+ext+field+'_99.npy',value)

quit()
value = np.array(df['sat_type'].values)
print(value[:10])
value = value.astype(int)
np.save(SAM_dir+'sat_type_99.npy',value)
value = np.load(SAM_dir+'sat_type_99.npy')
print(value[:10])
value = df['halo_index'].values
value = value.astype(int)
print(value[:10])
np.save(SAM_dir+'hosthaloid_99.npy',value)

# restore np.load for future normal usage
np.load = np_load_old
quit()

f = h5py.File(SAM_dir+typ+'prop/'+typ+'prop_99.h5','r')['df']

h = 0.6774
k = 99

blocks = ['axis0', 'axis1', 'block0_items', 'block0_values', 'block1_items', 'block1_values']

fields = ['hosthaloid','roothaloid','redshift','mhalo','rhalo','mstar','x_position','y_position','z_position','m_vir']

h_fields = ['mhalo','rhalo','mstar','m_vir','x_position','y_position','z_position']

m_fields = ['mhalo','mstar','m_vir']

file_names = ['hosthaloid_%d.npy'%k,'roothaloid_%d.npy'%k,'redshift_%d.npy'%k,'mhalo_%d.npy'%k,'rhalo_%d.npy'%k,'mstar_%d.npy'%k,'x_position_%d.npy'%k,'y_position_%d.npy'%k,'z_position_%d.npy'%k]


b0_i = blocks[2]
b0_v = blocks[3]
b1_i = blocks[4]
b1_v = blocks[5] 

'''
for i, item in enumerate(f[b0_i]):
    item = item.decode('UTF-8')
    print(item)
    if item in fields:
        print("we were looking for you")
        
        value = f[b0_v][:,i]
        if item in h_fields:
            value *= h
            print("h_field")
            if item in m_fields:
                value *= 1.e9
                print("m_field")
            
        f_name = item+'_99.npy'
        print(f_name)
        if typ == 'halo': np.save(SAM_dir+typ+'_'+f_name,value);
        else: np.save(SAM_dir+f_name,value)
        print("=======================")
    else:
        print("=========================")
'''
for i, item in enumerate(f[b1_i]):
    item = item.decode('UTF-8')
    print(item)
    if item in fields:
        print("we were looking for you")
        print(f[b1_v].shape)
        value = f[b1_v][:,i]
        if item in h_fields:
            value *= h
            print("h_field")
            if item in m_fields:
                value *= 1.e9
                print("m_field")
            
        f_name = item+'_99.npy'
        print(f_name)
        if typ == 'halo': np.save(SAM_dir+typ+'_'+f_name,value);
        else: np.save(SAM_dir+f_name,value)
        print("=======================")
    else:
        print("=========================")

        
'''
# 0 hosthaloid (long long)
# 1 birthhaloid (long long)
# 2 roothaloid (long long)
# 3 redshift
# 4 sat_type 0= central
# 5 mhalo total halo mass [1.0E09 Msun]
# 6 m_strip stripped mass [1.0E09 Msun]
# 7 rhalo halo virial radius [Mpc)]
# 8 mstar stellar mass [1.0E09 Msun]
# 9 mbulge stellar mass of bulge [1.0E09 Msun] 
# 10 mstar_merge stars entering via mergers] [1.0E09 Msun]
# 11 v_disk rotation velocity of disk [km/s] 
# 12 sigma_bulge velocity dispersion of bulge [km/s]
# 13 r_disk exponential scale radius of stars+gas disk [kpc] 
# 14 r_bulge 3D effective radius of bulge [kpc]
# 15 mcold cold gas mass [1.0E09 Msun]
# 16 mHI cold gas mass [1.0E09 Msun]
# 17 mH2 cold gas mass [1.0E09 Msun]
# 18 mHII cold gas mass [1.0E09 Msun]
# 19 Metal_star metal mass in stars [Zsun*Msun]
# 20 Metal_cold metal mass in cold gas [Zsun*Msun] 
# 21 sfr instantaneous SFR [Msun/yr]
# 22 sfrave20myr SFR averaged over 20 Myr [Msun/yr]
# 23 sfrave100myr SFR averaged over 100 Myr [Msun/yr]
# 24 sfrave1gyr SFR averaged over 1 Gyr [Msun/yr]
# 25 mass_outflow_rate [Msun/yr]
# 26 metal_outflow_rate [Msun/yr]
# 27 mBH black hole mass [1.0E09 Msun]
# 28 maccdot accretion rate onto BH [Msun/yr]
# 29 maccdot_radio accretion rate in radio mode [Msun/yr]
# 30 tmerge time since last merger [Gyr] 
# 31 tmajmerge time since last major merger [Gyr]
# 32 mu_merge mass ratio of last merger []
# 33 t_sat time since galaxy became a satellite in this halo [Gyr]
# 34 r_fric distance from halo center [Mpc]
# 35 x_position x coordinate [cMpc]
# 36 y_position y coordinate [cMpc]
# 37 z_position z coordinate [cMpc]
# 38 vx x component of velocity [km/s]
# 39 vy y component of velocity [km/s]
# 40 vz z component of velocity [km/s]
'''
