#!/usr/bin/env bash
kinit rmusters -k -t /home/cluster/keytab
#spark-submit --py-files /home/cluster/master/hadoop/lda.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --deploy-mode cluster --driver-memory 50g --executor-memory 20g --num-executors 10 /home/cluster/master/hadoop/lda_bikmeans.py
spark-submit --master yarn --deploy-mode cluster --num-executors 100 /home/cluster/master/hadoop/lda_bikmeans.py