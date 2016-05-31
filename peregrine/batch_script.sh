#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=w2vperegrine
#SBATCH --mem=2000
#SBATCH --partition=short
module load Python/3.4.2-goolfc-2.7.11
module load gensim
python /data/s1774395/master/peregrine/peregrine.py