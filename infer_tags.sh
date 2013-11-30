#!/bin/sh

#cd apriltag-0.91
#make
#cp tagtest ../
#cd ../


dir1='Data/nov_26_1'
dir2='Data/nov_26_2'
dir3='Data/nov_26_3'
dir=$dir3
files=`ls -1 ${dir} | grep .pnm`

for file in $files; do
    ./tagtest ${dir}/$file > "${dir}/${file%.pnm}.det"
done 
