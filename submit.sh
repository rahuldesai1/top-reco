#!/bin/bash

frac[1]=0.4
frac[2]=0.5
frac[3]=0.6

bs[1]=256
bs[2]=512

lr[1]=0.01
lr[2]=0.001
lr[3]=0.0001

opt[1]="tf.train.AdamOptimizer"

loss[1]="weighted"

dr[1]=0.4
dr[2]=0.5

for layers in `seq 7 7`
do
    for i in `seq 1 1`
    do
        batch_size=${bs[$i]}
        for j in `seq 1 1`
        do
            learning_rate=${lr[$j]}
            for o in `seq 1 1`
            do 
                optimizer=${opt[$o]}
                for l in `seq 1 1`
                do
                    loss_function=${loss[$l]}
                    for d in `seq 1 1`
                    do
                        dropout=${dr[$d]}
                        for f in `seq 1 1`
                        do 
                            sample_fraction=${frac[$f]}
                            echo $sample_fraction $layers $batch_size $optimizer $learning_rate $loss_function $dropout
                            sbatch ~/projects/train.sh $sample_fraction $layers $batch_size $optimizer $learning_rate $loss_function $dropout
            done
            done
        done
        done
    done
    done
done

