#!/usr/bin/env bash
kinit rmusters -k -t /home/cluster/keytab
#spark-submit --master yarn --deploy-mode cluster --driver-memory 50g --executor-memory 10g --num-executors 89 /home/cluster/master/hadoop/kmeans.py
#spark-submit --master yarn --deploy-mode cluster --driver-memory 10g --executor-memory 2g --num-executors 20 /home/cluster/master/hadoop/kmeans.py
spark-submit --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --deploy-mode cluster --driver-memory 10g --executor-memory 2g --num-executors 300 /home/cluster/master/hadoop/kmeans.py
