#!/usr/bin/env bash
1="w2v"
declare -a arr=(111 140 16 189 190 231 235 253 263 340 353 362 366 412 433 441 458 497 8)
for i in "${arr[@]}"

do
if [ $1 = "w2v" ]; then
cluster="$i"
command="hdfs dfs -getmerge "
fname="w2v_data_cluster_"
space=" "
local="/home/cluster/"
end=".csv"
newcommand=$command$fname$cluster$end$space$local$fname$cluster$end
echo $newcommand
$newcommand
fi

if [ $1 = "lda" ]; then
cluster="$i"
command="hdfs dfs -getmerge "
fname="lda_data_jan_cluster_merged_"
space=" "
local="/home/cluster/"
end=".csv"
newcommand=$command$fname$cluster$end$space$local$fname$cluster$end
echo $newcommand
$newcommand
fi

done