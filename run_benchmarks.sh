#!/bin/bash

mkdir benchmark-results

cat /proc/cpuinfo > benchmark-results/cpuinfo.txt

for x in 1 2 4 8
do
    for y in 1 2 4 8
    do 
        echo "Benchmarking tilesize ${x}x${y} with synthetic benchmark"
        #make clean
        #TILESIZE_X=$x TILESIZE_Y=$y make benchmark 2>&1 | tee benchmark-results/benchmark-threads-1-tilesize-${x}x${y}.txt
    done
done

for x in 1 2 4 8
do
    for y in 1 2 4 8
    do 
        echo "Benchmarking tilesize ${x}x${y} with llama main"
        make clean
        TILESIZE_X=$x TILESIZE_Y=$y make benchmark_main 2>&1 | tee benchmark-results/benchmark-main-threads-2-tilesize-${x}x${y}.txt
    done
done

MACHINE_ID=$(cat cat /etc/machine-id)
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

tar -czvf benchmark-results-$MACHINE_ID-$TIMESTAMP.tgz benchmark-results/*

echo "Done creating benchmark-results-$MACHINE_ID-$TIMESTAMP.tgz"