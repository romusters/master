


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
	assert(filter.filter("hoi") == "hoi")
	assert (filter.filter("2016") == "2016")
	assert (filter.filter(" ") == "")
	assert (filter.filter("http://www.cognata.nl") == "<url>")
	assert (filter.filter("werkte") == "werkt")
	assert (filter.filter("0655954252") == "<mobiel>")
	assert (filter.filter("065595") == "65595")
	assert (filter.filter(" 00 ") == "0")
	assert (filter.filter("#test") == "test")
	assert (filter.filter("@robert") == "<mention>")
	assert (filter.filter("robert.") == "robert")
	assert (filter.filter(" a ") == "a")
	# assert (filter.filter("01a01") == "a")
	print filter.filter(" #0008 0024horlogesjaargang2014 01a01 hoi  ik ben robert en ik werkte in 2016 00 test aan 0655954352 http://www.cognata.nl 01 0010 010920")
	print filter.filter(" a ")