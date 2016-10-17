#!/usr/bin/env bash
kinit rmusters -k -t /home/cluster/keytab

spark-submit --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --deploy-mode cluster --driver-memory 10g --executor-memory 2g --num-executors 65 /home/cluster/master/hadoop/skewness.py
