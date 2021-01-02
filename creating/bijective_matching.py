import numpy as np
import matplotlib.pyplot as plt

#root = '/mnt/gosling1/boryanah/TNG100/'
root = '/mnt/gosling1/boryanah/TNG300/'

#fp_dmo_inds_sub = np.load(root+'fp_dmo_inds_sub.npy').astype(int)
fp_dmo_inds_sub = np.load(root+'sub_match_fp_dmo_2500.npy').astype(int)
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp.npy').astype(int)
SubhaloGrNr_dm = np.load(root+'SubhaloGrNr_dm.npy').astype(int)
GroupFirst_fp = np.load(root+'GroupFirstSub_fp.npy').astype(int)
GroupFirst_dm = np.load(root+'GroupFirstSub_dm.npy').astype(int)
GroupPos_fp = np.load(root+'GroupPos_fp.npy')
GroupPos_dm = np.load(root+'GroupPos_dm.npy')
fp_inds_sub = fp_dmo_inds_sub[0]
dmo_inds_sub = fp_dmo_inds_sub[1]

matches = len(fp_inds_sub)
print(matches)

# which of the subhalos that have matches in FP are firsts
first_fp,comm1,comm2 = np.intersect1d(fp_inds_sub,GroupFirst_fp,return_indices=True)
#first_fp,comm1,comm2 = np.intersect1d(fp_inds_sub,np.where(SubhaloParent_fp==0)[0],return_indices=True)
#first_fp = fp_inds_sub; comm1 = np.arange(matches,dtype=int)
# what are their parent indices
par_fp = SubhaloGrNr_fp[first_fp]
# which subhalos in DM do they point to
sub_dm = dmo_inds_sub[comm1]
# what are the parent indices of these DM subhalos
par_dm = SubhaloGrNr_dm[sub_dm]
# stack the one way matching
fp_to_dmo_halo_inds = np.vstack((par_fp,par_dm))

print(fp_to_dmo_halo_inds.shape)

# same thing but change fp for dm
# which of the subhalos that have matches in DM are firsts
first_dm,comm1,comm2 = np.intersect1d(dmo_inds_sub,GroupFirst_dm,return_indices=True)
#first_dm,comm1,comm2 = np.intersect1d(dmo_inds_sub,np.where(SubhaloParent_dm==0)[0],return_indices=True)
#first_dm = dmo_inds_sub; comm1 = np.arange(matches,dtype=int)
# what are their parent indices
par_dm = SubhaloGrNr_dm[first_dm]
# which subhalos in FP do they point to
sub_fp = fp_inds_sub[comm1]
# what are the parent indices of these FP subhalos
par_fp = SubhaloGrNr_fp[sub_fp]
# stack the one way matching
dmo_to_fp_halo_inds = np.vstack((par_dm,par_fp))

print(dmo_to_fp_halo_inds.shape)

# out of the matched FP halos which are overlapping in both directions
fp_halo, comm1, comm2 = np.intersect1d(fp_to_dmo_halo_inds[0],dmo_to_fp_halo_inds[1],return_indices=True)
# for both directions, of the overlapping FP halos what are their DM halo equivalents
print(len(comm1))
dmo_halo_fp_to_dmo = (fp_to_dmo_halo_inds[1])[comm1]
dmo_halo_dmo_to_fp = (fp_to_dmo_halo_inds[0])[comm2]
dmo_halo_inds,comm1,comm2 = np.intersect1d(dmo_halo_fp_to_dmo,dmo_halo_dmo_to_fp,return_indices=1)
fp_halo_inds = fp_halo[comm1]

# these equivalents should now be aligned; which of them are the same (cause the above may point to diff DM halos)
#fp_halo_inds = fp_halo[dmo_halo_fp_to_dmo==dmo_halo_dmo_to_fp]
#dmo_halo_inds = dmo_halo_fp_to_dmo[dmo_halo_fp_to_dmo==dmo_halo_dmo_to_fp]
print(len(dmo_halo_inds))
print(GroupPos_fp[fp_halo_inds][:20])
print(GroupPos_dm[dmo_halo_inds][:20])
#dmo_halo_inds, comm1, comm2 = np.intersect1d(dm_halo_fp_to_dmo,dm_halo_dmo_to_fp,return_indices=True)
#fp_halo_inds = fp_halo[]
fp_dmo_halo_inds = np.vstack((fp_halo_inds,dmo_halo_inds))
# TESTING
root = ''
np.save(root+"fp_dmo_halo_inds.npy",fp_dmo_halo_inds)
plt.plot(np.arange(75000),np.arange(75000),'k--')
plt.scatter(GroupPos_fp[fp_halo_inds,0],GroupPos_dm[dmo_halo_inds,0],s=1)
plt.scatter(GroupPos_fp[fp_halo_inds,1],GroupPos_dm[dmo_halo_inds,1],s=1)
plt.show()
