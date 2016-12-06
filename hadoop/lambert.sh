#!/usr/bin/env bash
kinit rmusters -k -t /home/cluster/keytab
spark-submit --master yarn --deploy-mode cluster --num-executors 40 hadoop/lambert.py