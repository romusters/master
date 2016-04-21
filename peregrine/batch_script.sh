#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --job-name=w2v2gb
#SBATCH --mem=8000
#SBATCH --partition=short
module load Python/3.4.2-goolfc-2.7.11
module load gensim
python /data/s1774395/code/peregrine.py