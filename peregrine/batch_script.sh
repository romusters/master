#!/bin/bash
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --job-name=w2vDistances
#SBATCH --mem=80000
#SBATCH --partition=nodes
module load Python/3.4.2-goolfc-2.7.11
module load gensim
python /data/s1774395/master/hadoop/vectors.py
