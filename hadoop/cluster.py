from pyspark import SparkContext, SparkConf, SQLContext
import logging, sys
import numpy as np

# spark-submit --packages com.databricks:spark-csv_2.10:1.4.0 --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --master yarn --deploy-mode cluster  master/hadoop/cluster.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

conf = SparkConf()\
    .set("spark.driver.maxResultSize", "0")\
	.set("spark.driver.memory", "12g")\
	.set("spark.executor.memory", "12g")\
	.set("spark.executor.instances", "75")\
	.set("spark.executor.cores", "8")

sc = SparkContext(appName='cluster', conf=conf)


#load the word2vec model
def load_w2v_model():
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	lookup = sqlContext.read.parquet('/user/rmusters/2015model99/data').alias("lookup")
	lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())
	return lookup_bd

#load lda model
def load_model():
	from pyspark.mllib.clustering import LDA, LDAModel
	#sc = SparkContext(appName='lda_load', conf=conf)
	path = "/user/rmusters/ldaModel2"
	ldaModel = LDAModel.load(sc, path)
	return ldaModel

#load the data
def load_data(path):
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet(path)
	return data



def cos_similarity(tweet, other_tweet, model):
	import numpy
	sims = []
	for word in tweet:
		tmp_cos_sim = 0
		try:
			word = model.value.get(word)
		except TypeError:
			continue
		for other_word in other_tweet:
			other_word = model.value.get(other_word)
			#cos_sim = numpy.dot(model.transform(word), model.transform(other_word)) / (numpy.linalg.norm(model.transform(word)) * numpy.linalg.norm(model.transform(other_word)))
			try:
				cos_sim = numpy.dot(word, other_word) / (numpy.linalg.norm(word) * numpy.linalg.norm(other_word))
			except TypeError:
				continue
			if cos_sim > tmp_cos_sim:
				tmp_cos_sim = cos_sim
		sims.append(tmp_cos_sim)
	return float(sum(sims)/len(sims))

def save_data_sample():
	data = load_data('hdfs:///user/rmusters/data')
	data = data.sample(False, 0.01)
	data.write.parquet("hdfs:///user/rmusters/data_sample", mode="overwrite")

def save_data_sample_csv():
	data = load_data('hdfs:///user/rmusters/data_sample')
	data.save("data_sample.csv", "com.databricks.spark.csv", "overwrite")

def main():
	path = 'hdfs:///user/rmusters/data'
	path = 'hdfs:///user/rmusters/data_sample'
	#write_data(path)

	#load filtered tweets
	data = load_data(path)

	#load w2v model
	model = load_w2v_model()

	tweet = data.sample(False, 0.1, seed=0).limit(1).select("filtered_text").collect()[0][0]
	#calculate distances from a tweet to the others.
	data_rdd = data.rdd.map(lambda (text, filtered_text, vectors, id): (text, filtered_text, vectors, id, cos_similarity(tweet, text, model)))

	#
	logging.info(data_rdd.count())

	cluster = data_rdd.filter(lambda (text, filtered_text, vectors, id, similarity): similarity > 0.6)
	logging.info(cluster.count())



if __name__ == "__main__":
	#main()
	save_data_sample()