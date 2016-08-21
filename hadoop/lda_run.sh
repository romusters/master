#!/usr/bin/env bash
kinit rmusters -k -t /home/cluster/keytab
spark-submit --py-files /home/cluster/master/hadoop/lda.py --master yarn --deploy-mode cluster --driver-memory 50g --executor-memory 10g --num-executors 89 /home/cluster/master/hadoop/lda_kmeans.py