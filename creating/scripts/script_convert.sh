#convert -delay 10 -loop 1  time_dm_gas_low_1340_*png animated_1340.gif;

input_low="/home/boryanah/SAM/SAM_TNG_clustering/creating/visuals/inds_low.txt"
input_high="/home/boryanah/SAM/SAM_TNG_clustering/creating/visuals/inds_high.txt"
output="convert.sh"

rm "$output"

while IFS= read -r line
do
    echo -n "convert -delay 10 -loop 1 time_dm_gas_low_${line}_*png animated_low_$line.gif" >> "$output"
    echo "" >> "$output"
done < "$input_low"

while IFS= read -r line
do
    echo -n "convert -delay 10 -loop 1 time_dm_gas_high_${line}_*png animated_high_$line.gif" >> "$output"
    echo "" >> "$output"
done < "$input_high"
