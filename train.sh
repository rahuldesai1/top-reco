#!/usr/bin/env bash
#rm tarlog
#./tar.sh >> tarlog
#SBATCH -C haswell
#SBATCH -q shared

# hyper-parameters
hyp1=$1
hyp2=$2
hyp3=$3
hyp4=$4
hyp5=$5
hyp6=$6
hyp7=$7

mkdir run
rundir="~/projects/run/$hyp1$hyp2$hyp3$hyp4$hyp5$hyp6$hyp7"
rm -rf $rundir
mkdir $rundir
cd $rundir

echo $PWD
# Do this in $SLURM_TMP if in batch mode
# Otherwise make a run subdirectory in the submission dir

rm progress.txt
echo "1--setup" >> progress.txt
source ~/miniconda2/bin/activate py3.6

# Do the training
echo "2--begin train" >> progress.txt
python3 ~/projects/train.py $hyp1 $hyp2 $hyp3 $hyp4 $hyp5 $hyp6 $hyp7
echo "3--end train" >> progress.txt
conda deactivate
