#!/bin/sh
# Want a full node
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 1:00:00
#SBATCH --qos bearchal0x
#SBATCH --reservation BearChallenge

module purge; module load bluebear

module load bear-apps/2018a
module load TensorFlow/1.8.0-foss-2018a-Python-3.6.3

sh train-network.sh -d mnist/ -e 1

