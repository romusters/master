#!/bin/bash
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=w2vFullAttempt3
#SBATCH --mem=500000
#SBATCH --partition=himem
module load Python/3.4.2-goolfc-2.7.11
module load gensim
python /data/s1774395/master/peregrine/peregrine.py