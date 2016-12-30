#!/usr/bin/env bash
kinit rmusters -k -t /home/cluster/keytab
spark-submit --master yarn --deploy-mode cluster --packages com.databricks:spark-csv_2.10:1.4.0 --num-executors 10 lambert.py