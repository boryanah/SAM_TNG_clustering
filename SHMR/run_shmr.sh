./shmr.py --num_gals 1200 --type_gal mhalo --want_matching_hydro --want_matching_sam
./shmr.py --num_gals 6000 --type_gal mhalo --want_matching_hydro --want_matching_sam
./shmr.py --num_gals 12000 --type_gal mhalo --want_matching_hydro --want_matching_sam

./shmr.py --num_gals 6000 --type_gal mhalo --secondary_property env --want_matching_hydro --want_matching_sam
./shmr.py --num_gals 6000 --type_gal mhalo --secondary_property rvir --want_matching_hydro --want_matching_sam
./shmr.py --num_gals 6000 --type_gal mhalo --secondary_property conc --want_matching_hydro --want_matching_sam
./shmr.py --num_gals 6000 --type_gal mhalo --secondary_property vdisp --want_matching_hydro --want_matching_sam
./shmr.py --num_gals 6000 --type_gal mhalo --secondary_property spin --want_matching_hydro --want_matching_sam
./shmr.py --num_gals 6000 --type_gal mhalo --secondary_property s2r --want_matching_hydro --want_matching_sam

./shmr.py --num_gals 1200 --type_gal mstar
./shmr.py --num_gals 6000 --type_gal mstar
./shmr.py --num_gals 12000 --type_gal mstar

./shmr.py --num_gals 1200 --type_gal sfr
./shmr.py --num_gals 6000 --type_gal sfr
./shmr.py --num_gals 12000 --type_gal sfr

./shmr.py --num_gals 6000 --type_gal mstar --secondary_property env
./shmr.py --num_gals 6000 --type_gal mstar --secondary_property rvir
./shmr.py --num_gals 6000 --type_gal mstar --secondary_property conc
./shmr.py --num_gals 6000 --type_gal mstar --secondary_property vdisp
./shmr.py --num_gals 6000 --type_gal mstar --secondary_property spin
./shmr.py --num_gals 6000 --type_gal mstar --secondary_property s2r