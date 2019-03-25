#!/usr/bin/env bash
#rm tarlog
#./tar.sh >> tarlog
#SBATCH -C haswell

here=$PWD

# hyper-parameters
hyp1=$1
hyp2=$2
hyp3=$3
hyp4=$4
hyp5=$5
hyp6=$6
hyp7=$7

# Do this in $SLURM_TMP if in batch mode
# Otherwise make a run subdirectory in the submission dir
echo $here
rundir1=$SLURM_TMP
mkdir ${here}/run
if [ "$SLURM_TMP" == "" ]; then rundir1=${here}/run; fi
cd ${rundir1}

rm progress.txt
echo "1--setup" >> progress.txt
source ~/miniconda2/bin/activate py3.6

# Do the training
echo "2--begin train" >> progress.txt
python3 ~/projects/train.py $hyp1 $hyp2 $hyp3 $hyp4 $hyp5 $hyp6 $hyp7
echo "3--end train" >> progress.txt
conda deactivate
