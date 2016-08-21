#!/usr/bin/env bash
kinit rmusters -k -t /home/cluster/keytab
spark-submit --master yarn --deploy-mode cluster --driver-memory 50g --executor-memory 10g --num-executors 189 /home/cluster/master/hadoop/w2v_kmeans.py