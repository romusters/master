


import logging, sys
import filter

#spark-submit --py-files hadoop/stemmer.py,hadoop/filter.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --executor-memory 20g --deploy-mode cluster  hadoop/stemmer_test.py
#spark-submit --master yarn --deploy-mode cluster --packages com.databricks:spark-csv_2.10:1.4.0 --driver-memory 50g --executor-memory 10g --num-executors 100 w2v.py

#if in yarn mode, all cores are used
remote = 0
if remote:
	from pyspark import SparkContext, SparkConf
	from pyspark.mllib.feature import Word2Vec, HashingTF
	from pyspark.sql import SQLContext
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
	logger = logging.getLogger(__name__)
	sc = SparkContext(appName='Stemmer test')
	sqlContext = SQLContext(sc)
	loc = '/user/rmusters/text/2015/01/20150101-00.out.gz.txt'
	text_file = sc.textFile(loc)
	data = text_file.map(lambda line: filter.filter(line))
	data = data.map(lambda x: (x, )).toDF()
	data.write.format("com.databricks.spark.csv").mode("overwrite").save("data_test.csv")
	print data.show()
	text_file = text_file.map(lambda x: (x, )).toDF()
	print text_file.show()
else:
	print filter.filter("toegankelijk")