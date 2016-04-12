

import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME']="/home/robert/spark-1.6.1-bin-hadoop2.6"

# Append pyspark  to Python Path
sys.path.append("/home/robert/spark-1.6.1-bin-hadoop2.6/bin")

try:
	from pyspark import SparkContext
	from pyspark import SparkConf
	from pyspark import SparkContext
	from pyspark.mllib.feature import Word2Vec, Word2VecModel
	print ("Successfully imported Spark Modules")

except ImportError as e:
	print ("Can not import Spark Modules", e)
	sys.exit(1)

import os
dir = '/home/robert/data/20151225to31/tar/'
filenames = os.listdir(dir)

word2vec = Word2Vec()
sc = SparkContext(appName='Word2Vec')
model = None
idx = 0
for filename in filenames:
	print dir + filename
	os.rename(dir + filename, dir + filename.replace(':', ''))

	inp = sc.textFile(dir + filename).map(lambda row: row.split(" "))

	if idx == 0:
		model = word2vec.fit(inp)
	else:
		model =  Word2VecModel.load(sc, "/home/robert/data/w2vmodels/hadoop" + str(idx - 1))
		model.train(inp)

	model.save(sc, "/home/robert/data/w2vmodels/hadoop" + str(idx))
	idx += 1

