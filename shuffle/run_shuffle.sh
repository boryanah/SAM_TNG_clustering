./shuffle.py --num_gals 1200 --type_gal mhalo --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 12000 --type_gal mhalo --want_matching_sam --want_matching_hydro

./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  env --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  rvir --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  conc --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  conc_asc --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  conc_desc --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  vdisp --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  vdisp_desc --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  s2r_desc --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  s2r_asc --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  spin_desc --want_matching_sam --want_matching_hydro
./shuffle.py --num_gals 6000 --type_gal mhalo --secondary_property  spin_asc --want_matching_sam --want_matching_hydro

./shuffle.py --num_gals 1200 --type_gal mstar
./shuffle.py --num_gals 6000 --type_gal mstar
./shuffle.py --num_gals 12000 --type_gal mstar

./shuffle.py --num_gals 1200 --type_gal sfr
./shuffle.py --num_gals 6000 --type_gal sfr
./shuffle.py --num_gals 12000 --type_gal sfr 

./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property env
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property rvir
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property conc
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property conc_asc
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property conc_desc
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property vdisp
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property vdisp_desc
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property s2r_desc
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property s2r_asc
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property spin_desc
./shuffle.py --num_gals 6000 --type_gal mstar --secondary_property spin_asc
