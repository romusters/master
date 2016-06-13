from pyspark import SparkContext, SparkConf
import sys
from pyspark.mllib.feature import Word2VecModel
from pyspark.mllib.feature import Word2Vec, HashingTF
import logging, sys

# spark-submit --master yarn --executor-memory 2g --driver-memory 2g --deploy-mode cluster --num-executors 4 master/hadoop/testLoadw2v.py

conf = (SparkConf()
		.set("spark.driver.maxResultSize", "0")
		# .set('spark.executor.memory','32g')
		# .set('spark.driver.memory','32g')
		)

sc = SparkContext(appName='saveFileTest', conf=conf)

loc = "/user/rmusters/testData"
inp = sc.textFile(loc).map(lambda line: line.split(" "))
# tweet = "Simple test tweet to test w2v.".split(" ")
# inp = sc.parallelize(tweet)

word2vec = Word2Vec()
word2vec.setMinCount(1)

model = word2vec.fit(inp)

model.save(sc, '/user/rmusters/testModel3')

#kan driver niet aan...
#vec = model.getVectors()
#print type(vec)
#vec = str(vec)
#rdd = sc.parallelize(vec)


#path =  'hdfs:///user/rmusters/testVectors.txt'
path = "/user/rmusters/testVectors3.txt"
rdd.saveAsTextFile(path)

