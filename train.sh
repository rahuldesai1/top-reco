#!/usr/bin/env bash
#rm tarlog
#./tar.sh >> tarlog
#SBATCH -C haswell
#SBATCH -q shared
#SBATCH -t 4:00:00

# hyper-parameters
hyp1=$1
hyp2=$2
hyp3=$3
hyp4=$4
hyp5=$5
hyp6=$6
hyp7=$7

mkdir run
rundir=/global/homes/r/rahuld/projects/run/$hyp1$hyp2$hyp3$hyp4$hyp5$hyp6$hyp7
rm -rf $rundir
mkdir $rundir
cd $rundir

echo $PWD

rm -f progress.txt
echo "1--setup" >> progress.txt
source /global/homes/r/rahuld/projects/miniconda2/bin/activate py3.6

# Do the training
echo "2--begin train" >> progress.txt
ls ~/projects/train.py
python3 /global/homes/r/rahuld/projects/train.py $hyp1 $hyp2 $hyp3 $hyp4 $hyp5 $hyp6 $hyp7
echo "3--end train" >> progress.txt
conda deactivate
