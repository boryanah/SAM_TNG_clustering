import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from correlation_function import get_corrfunc
from sam2tng import SAM2TNG
from sam2tng import TNGgal_index
from sam2tng import SAMgal_index

# Changeable
galaxy_num = 6000
hgals_ordering = "mass_gals_hydro.npy" # "SubhaloSFR_fp.npy"
sgals_ordering = "GalpropMstar.npy" # "GalpropSfr.npy"

def Step_One(hID_gals, matched_index, hcentre_matched, gals_pos):
    # og
    #galaxy_in_halo_count = np.zeros(matched_index.shape, dtype=int)
    galaxy_in_halo_count = np.zeros(hcentre_matched.shape[0], dtype=int)
    relative_pos = np.zeros(gals_pos.shape)
    centre_pos = np.zeros(gals_pos.shape)
    startindex = np.full(hcentre_matched.shape[0], -999, dtype=int)
    unique_galsID, gals_count = np.unique(hID_gals, return_counts=True)
    count = 0

    for i in range(unique_galsID.shape[0]):
        # B.H. remed
        halo_position_index = np.where(matched_index == unique_galsID[i])[0]
        # B.H. added
        #halo_position_index = unique_galsID[i]
        galaxy_positions_index = np.where(hID_gals == unique_galsID[i])[0]
        
        galaxy_in_halo_count[halo_position_index] = gals_count[i]
        startindex[halo_position_index] = count
        assert gals_count[i] == len(galaxy_positions_index), "problem with gal num"
        
        halo_centre = hcentre_matched[halo_position_index]
        
        centre_pos[count:count+gals_count[i]] = halo_centre
        # B.H. remed
        rel_pos = gals_pos[galaxy_positions_index] - centre_pos[count: count+gals_count[i]]
        dx = rel_pos[:,0]
        dy = rel_pos[:,1]
        dz = rel_pos[:,2]
        dx = np.where(np.abs(dx) > Lbox/2.,-np.sign(dx)*np.abs(np.abs(dx)-Lbox),dx)
        dy = np.where(np.abs(dy) > Lbox/2.,-np.sign(dy)*np.abs(np.abs(dy)-Lbox),dy)
        dz = np.where(np.abs(dz) > Lbox/2.,-np.sign(dz)*np.abs(np.abs(dz)-Lbox),dz)
        rel_pos[:,0] = dx
        rel_pos[:,1] = dy
        rel_pos[:,2] = dz

        #rel_pos[np.abs(dy) > Lbox/2.,1] = -np.sign(dy[np.abs(dy) > Lbox/2.])*np.abs((np.abs(dy[np.abs(dy) > Lbox/2.])-Lbox))
        relative_pos[count:count+gals_count[i]] = rel_pos
        # B.H. added
        #relative_pos[count:count+gals_count[i]] = np.where(np.abs(gals_pos[galaxy_positions_index] - halo_centre)>0.5*Lbox,gals_pos[galaxy_positions_index] - halo_centre,gals_pos[galaxy_positions_index] - halo_centre)
        count = count + gals_count[i]

    to_shuffle = np.zeros((hcentre_matched.shape[0], 2), dtype=int)
    to_shuffle[:,0] = galaxy_in_halo_count
    to_shuffle[:,1] = startindex
    diff = 30.
    print("number of objects that are super far = ",np.sum(np.abs(relative_pos[:,0])>diff))
    print("percentage of objects that are super far = ",np.sum(np.abs(relative_pos[:,0])>diff)*100./relative_pos.shape[0])
    print("number of objects that are super far = ",np.sum(np.abs(relative_pos[:,1])>diff))
    print("percentage of objects that are super far = ",np.sum(np.abs(relative_pos[:,1])>diff)*100./relative_pos.shape[0])
    print("number of objects that are super far = ",np.sum(np.abs(relative_pos[:,2])>diff))
    print("percentage of objects that are super far = ",np.sum(np.abs(relative_pos[:,2])>diff)*100./relative_pos.shape[0])
    print(np.max(np.abs(relative_pos),axis=1)[:100])
    print("number of objects that are super far = ",np.sum(np.max(np.abs(relative_pos),axis=1) > diff))
    print("percentage of objects that are super far = ",np.sum(np.max(np.abs(relative_pos),axis=1) > diff)*100./relative_pos.shape[0])
    print("number of objects that are super far = ",np.sum(np.abs(relative_pos) > diff))
    print("percentage of objects that are super far = ",np.sum(np.abs(relative_pos) > diff)*100./relative_pos.shape[0])
    
    print(np.max(np.abs(relative_pos)))
    '''
    print(np.sum(np.abs(relative_pos)))
    print(galaxy_in_halo_count[:100])
    print(startindex[:100])
    print(np.sum(galaxy_in_halo_count))
    print(np.min(relative_pos))
    print(np.max(relative_pos))
    print(np.mean(np.abs(relative_pos)))
    '''
    print(np.sum(galaxy_in_halo_count>0))
    return to_shuffle, centre_pos, relative_pos

def Step_Two(SAM_to_shuffle, TNG_to_shuffle):
    for i in range(bins_number):
        within_mass_range = np.logical_and(matched_mhalo >= bin_edges[i], matched_mhalo <= bin_edges[i+1])
        np.random.seed()
        SAM_to_shuffle[within_mass_range] = np.random.permutation(SAM_to_shuffle[within_mass_range])
        TNG_to_shuffle[within_mass_range] = np.random.permutation(TNG_to_shuffle[within_mass_range])
    return SAM_to_shuffle, TNG_to_shuffle

def Step_Three(shuffled, hcentre_matched, gals_pos):
    relative_pos = np.zeros(gals_pos.shape)
    centre_pos = np.zeros(gals_pos.shape)
    occupied = np.where(shuffled[:,1] != -999)[0]
    count = 0
    print(len(occupied))
    

    for i in range(occupied.shape[0]):
        gals_count = shuffled[occupied[i],0]
        gals_start = shuffled[occupied[i],1]
        halo_centre = hcentre_matched[occupied[i]]

        centre_pos[count:count+gals_count] = np.tile(halo_centre, (gals_count,1))
        #relative_pos[count:count+gals_count] = gals_pos[gals_start:gals_start+gals_count] + centre_pos[count:count+gals_count]
        relative_pos[count:count+gals_count] = gals_pos[gals_start:gals_start+gals_count] + halo_centre
        count = count + gals_count

    return centre_pos, relative_pos

def Step_Four(gals_pos, centre_pos, relative_pos, label):
    '''
    rel_cn = get_corrfunc(relative_pos, cn_bins, Lbox)
    # cen_cn = get_corrfunc(centre_pos, cn_bins, Lbox)
    pos_cn = get_corrfunc(gals_pos, cn_bins, Lbox)
    '''
    import Corrfunc
    pos_cn = Corrfunc.theory.xi(X=gals_pos[:,0],Y=gals_pos[:,1],Z=gals_pos[:,2],binfile=cn_bins,boxsize=Lbox,nthreads=16)['xi']
    rel_cn = Corrfunc.theory.xi(X=relative_pos[:,0],Y=relative_pos[:,1],Z=relative_pos[:,2],binfile=cn_bins,boxsize=Lbox,nthreads=16)['xi']
    plt.axhline(y=1, color='k', linewidth=3, linestyle=":")
    plt.xscale("log")
    plt.plot(cn_bins_centre, rel_cn/pos_cn, label=label)
    plt.ylim([0.4,1.5])
    plt.xlim([0.1,15])
    plt.legend()
    # plt.plot(cn_bins_centre, cen_cn/pos_cn)

# afraid // the neighbourhood lyrics
# Lunatic princess - bullet hell
# 8 - 23:08 - Reach for the Moon, Immortal Smoke (ft. AHMusic)
# Border of Life (04:04) - bullet hell

directorysam = '/mnt/store1/boryanah/SAM_subvolumes/'
directoryhydro = '/mnt/gosling1/boryanah/TNG100/'
directorysave = 'figs/'
Lbox = 75
# mass bins
bins_number = 30
bin_edges = np.logspace(10, 15, bins_number + 1)
# bins for corr_func
cn_bins = np.logspace(-1, 1, 21)
cn_bins_centre = 0.5*(cn_bins[1:] + cn_bins[:-1])

sat_type = np.load(directorysam + "GalpropSatType.npy")
mhalo = np.load(directorysam + "GalpropMhalo.npy")[sat_type == 0]
hgals_index = TNGgal_index(galaxy_num, hgals_ordering)
sgals_index = SAMgal_index(galaxy_num, sgals_ordering)
matched_mhalo, matchedTNGindex, matchedSAMindex = SAM2TNG(mhalo)

hhID_gals = np.load(directoryhydro + "SubhaloGrNr_fp.npy")[hgals_index]
hgals_pos = np.load(directoryhydro + "SubhaloPos_fp.npy")[hgals_index]/1000.
shID_gals = np.load(directorysam + "GalpropHaloIndex_corr.npy")[sgals_index]
sgals_pos = np.load(directorysam + "GalpropPos.npy")[sgals_index]
# Central halo positions
hhID = np.load(directoryhydro + "SubhaloGrNr_fp.npy")
# B.H. included h/shcentre
hhcentre = np.load(directoryhydro + "GroupPos_fp.npy")/1000.
hhcentre_matched = hhcentre[matchedTNGindex]
shID = np.load(directorysam + "GalpropHaloIndex_corr.npy")
shcentre = (np.load(directorysam + "GalpropPos.npy")[sat_type == 0])
shcentre_matched = shcentre[matchedSAMindex]

# B.H. remed
TNGshuffle, hcentre_pos, hrelative_pos = Step_One(hhID_gals, matchedTNGindex, hhcentre_matched, hgals_pos)
SAMshuffle, scentre_pos, srelative_pos = Step_One(shID_gals, matchedSAMindex, shcentre_matched, sgals_pos)

print("--------------------")
print(np.min(hrelative_pos))
print(np.min(srelative_pos))
print(np.max(hrelative_pos))
print(np.max(srelative_pos))

# B.H. added
#TNGshuffle, hcentre_pos, hrelative_pos = Step_One(hhID_gals, matchedTNGindex, hhcentre, hgals_pos)
#SAMshuffle, scentre_pos, srelative_pos = Step_One(shID_gals, matchedSAMindex, shcentre, sgals_pos)

# TESTING
quit()
SAMshuffled, TNGshuffled = Step_Two(SAMshuffle, TNGshuffle)
hcentre_pos, hrelative_pos = Step_Three(TNGshuffled, hhcentre_matched, hrelative_pos)
scentre_pos, srelative_pos = Step_Three(SAMshuffled, shcentre_matched, srelative_pos)

print("--------------------")
print(np.min(hrelative_pos))
print(np.min(srelative_pos))
print(np.max(hrelative_pos))
print(np.max(srelative_pos))
print(np.max(scentre_pos))
print(np.min(scentre_pos))
print(np.max(hcentre_pos))
print(np.min(hcentre_pos))

hrelative_pos[hrelative_pos < 0.] += Lbox
hrelative_pos[hrelative_pos >= Lbox] -= Lbox
srelative_pos[srelative_pos < 0.] += Lbox
srelative_pos[srelative_pos >= Lbox] -= Lbox
scentre_pos[scentre_pos < 0.] += Lbox
scentre_pos[scentre_pos >= Lbox] -= Lbox
sgals_pos[sgals_pos < 0.] += Lbox
sgals_pos[sgals_pos >= Lbox] -= Lbox
hcentre_pos[hcentre_pos < 0.] += Lbox
hcentre_pos[hcentre_pos >= Lbox] -= Lbox

print("--------------------")
print(np.min(hrelative_pos))
print(np.min(srelative_pos))
print(np.max(hrelative_pos))
print(np.max(srelative_pos))
print(np.max(scentre_pos))
print(np.min(scentre_pos))
print(np.max(hcentre_pos))
print(np.min(hcentre_pos))

print("ALMOST THERE")
Step_Four(hgals_pos, hcentre_pos, hrelative_pos, label='Hydro')
Step_Four(sgals_pos, scentre_pos, srelative_pos, label='SAM')
plt.savefig("figs/shuffle.png")
plt.show()
