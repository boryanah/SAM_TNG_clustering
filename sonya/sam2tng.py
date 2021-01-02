import numpy as np

def SAM2TNG(SAMarray):
    # Requirements: the SAM array must be of the type Haloprop, if it is not, use [sat_type == 0], 
    # where sat_type = np.load(directorysam + "GalpropSatType.npy")
    SubfindID = np.load(directorysam + "HalopropSubfindID.npy")
    matchedSAMindex = np.where(SubfindID != -1)[0]
    matchedarray = SAMarray[matchedSAMindex]

    TNGmatched = (SubfindID[SubfindID != -1])
    TNGSubHaloID = np.load(directoryhydro + "SubhaloGrNr_fp.npy")
    TNGmhalo = np.load(directoryhydro + "Group_M_Crit200_fp.npy") * 10**10
    TNG_index = np.arange(0,TNGmhalo.shape[0], dtype=int)
    matchedTNGindex = TNG_index[TNGSubHaloID[TNGmatched]]

    return matchedarray, matchedTNGindex, matchedSAMindex

def TNGgal_index(gals_number, sorting_factor):

    # Data
    SubfindID = np.load(directorysam + "HalopropSubfindID.npy")
    TNGSubhalos = np.load(directoryhydro + "SubhaloGrNr_fp.npy")
    TNGgals_index = np.argsort(np.load(directoryhydro + sorting_factor))[::-1][:gals_number]
    TNGgals = TNGSubhalos[TNGgals_index]

    # assuming that all the galaxies in the halo which have been matched can be safety kept.
    matchedTNGhalos = TNGSubhalos[SubfindID[SubfindID != -1]]
    boolean = np.in1d(TNGgals, matchedTNGhalos)
    matchedTNGgals_index = TNGgals_index[np.where(boolean == True)[0]]
    return matchedTNGgals_index

def SAMgal_index(gals_number, sorting_factor):

    # Data
    SubfindID = np.load(directorysam + "HalopropSubfindID.npy")
    SAMgals_index = np.argsort(np.load(directorysam + sorting_factor))[::-1][:gals_number]
    SAMgalsID = np.load(directorysam + "GalpropHaloIndex_corr.npy")[SAMgals_index]
    matched = np.where(SubfindID != -1)[0]
    SAMHaloID = np.load(directorysam + "HalopropIndex_corr.npy")[matched]
    boolean = np.in1d(SAMgalsID, SAMHaloID)
    matchedSAMgals_index = SAMgals_index[np.where(boolean == True)[0]]
    return matchedSAMgals_index

directorysam = '/mnt/store1/boryanah/SAM_subvolumes/'
directoryhydro = '/mnt/gosling1/boryanah/TNG100/'
