#!/bin/bash

frac[1]=0.3
frac[2]=0.4
frac[3]=0.5

bs[1]=128
bs[2]=256
bs[3]=512

lr[1]=0.1
lr[2]=0.01
lr[3]=0.001

opt[1]="tf.train.AdamOptimizer"
opt[2]="tf.train.GradientDescentOptimizer"
opt[3]="tf.train.RMSPropOptimizer"

loss[1]="weighted"
loss[2]="un_weighted"

dr[1]=0.5
dr[2]=0.4

for layers in `seq 7 12`
do
    for i in `seq 1 3`
    do
        batch_size=${bs[$i]}
        for j in `seq 1 3`
        do
            learning_rate=${lr[$j]}
            for o in `seq 1 3`
            do 
                optimizer=${opt[$o]}
                for l in `seq 1 2`
                do
                    loss_function=${loss[$l]}
                    for d in `seq 1 2`
                    do
                        dropout=${dr[$d]}
                        for f in `seq 1 3`
                        do 
                            sample_fraction=${frac[$f]}
                            echo $sample_fraction $layers $batch_size $optimizer $learning_rate $loss_function $dropout
                            sbatch train.sh $sample_fraction $layers $batch_size $optimizer $learning_rate $loss_function $dropout
            done
            done
        done
        done
    done
    done
done
