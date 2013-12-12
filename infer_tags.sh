#!/bin/bash

#cd apriltag-0.91
#make
#cp tagtest ../
#cd ../


dir1='Data/nov_26_1c'
dir2='Data/nov_26_2c'
dir3='Data/nov_26_3c'
dir4='Data/dec_7_1'
dir5='Data/dec_7_2'
dir6='Data/dec_7_3'
dir=$dir6
files=`ls -1 ${dir} | grep .pnm`

for file in $files; do
    ./tagtest ${dir}/$file > "${dir}/${file%.pnm}.det"
done 


# hmmpref='Data/hmm_train'
# cards=( E N S W X )
# FBs=( B F X ) 


# for ((i=0;i<5;i++))
# do
#     for((j=0;j<3;j++))
#     do
# 	dir=${hmmpref}/${cards[i]}/${FBs[j]}
# 	echo $dir
# 	files=`ls -1 ${dir} | grep .pnm`
# 	for file in $files; do
# 	    ./tagtest ${dir}/$file > "${dir}/${file%.pnm}.det"
# #	    convert "$file" "${file%.jpg}.pnm"
# 	done

#     done
# done