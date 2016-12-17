#!/usr/bin/env bash
kinit rmusters -k -t /home/cluster/keytab
spark-submit --master yarn --deploy-mode cluster  --packages com.databricks:spark-csv_2.10:1.4.0 --driver-memory 50g --executor-memory 10g --num-executors 10 /home/cluster/master/hadoop/w2v_kmeans.py