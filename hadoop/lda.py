from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

#spark-submit --master yarn --deploy-mode cluster master/hadoop/lda.py

conf = (SparkConf()
		.set("spark.driver.maxResultSize", "0") \
		.set("driver-memory", "20g") \
		.set("executor-memory", "20g") \
		.set("num-executors", "400"))

# Larger file
loc = '/user/rmusters/text/2015/*/*'




def train_model():
	sc = SparkContext(appName='lda_train', conf=conf)
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)

	model_vectors = sqlContext.read.parquet('/user/rmusters/2015model99/data')
	logger.info("model loaded")
	rdd_words = model_vectors.map(lambda line: line[0])
	words = rdd_words.collect() #15919
	logger.info("words collected")
	#logger.info(len(words))

	def load_data(path):
		sqlContext = SQLContext(sc)
		data = sqlContext.read.parquet(path)
		return data

	path = 'hdfs:///user/rmusters/data'
	data = load_data(path)
	logger.info("data loaded")
	data = data.sample(False, 0.01)
	logger.info("data sampled")
	#test = data.select("vectors").collect()
	#logger.info(len(test[0]))

	data = data.drop("vectors")

	def bow(filtered_text):
		vector = [0]*len(words)
		for i, w in enumerate(words):
			if w in filtered_text:
				vector[i] = vector[i]+1
		v_dict = {}
		for i, v in enumerate(vector):
			if v >= 1:
				v_dict[i] = v
			else:
				v_dict[i] = 0 # don't forget this or else the vocabsize and vector size will differ.
		return v_dict

	#check if sum of vector is zero 13 times. This indicates the datasample does not contain certain words and thus the sparse vector removes them

	from pyspark.mllib.linalg import SparseVector
	size = len(words)
	data = data.map(lambda (text, filtered_text, id): (text, filtered_text, SparseVector(size, bow(filtered_text)), id))
	logger.info("bag of words data")
	#could it be that the densevector reduce the vocabsize?
	vectors = data.toDF(["text", "filtered_text", "vectors", "id"]).select("vectors").collect()
	logger.info(len(vectors[0]))

	corpus = data.map(lambda (text, filtered_text, vector, id): [id, vector])
	corpus_vectors = corpus.toDF(["idx", "vector"]).select("vector").collect()
	logger.info(len(corpus_vectors[0]))

	ldaModel = LDA.train(corpus, k=100)
	logger.info(ldaModel.vocabSize())
	logger.info(ldaModel.describeTopics())
	ldaModel.save(sc, 'hdfs:///user/rmusters/ldaModel')

def load_model():
	from pyspark.mllib.clustering import LDA, LDAModel
	sc = SparkContext(appName='lda_load', conf=conf)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	path = "/user/rmusters/ldaModel"
	ldaModel = LDAModel.load(sc, path)
	return ldaModel

def predict():
	ldaModel = load_model()

	topics = ldaModel.describeTopics()

	topics = ldaModel.topicsMatrix()
	for topic in range(100):
		print "Topic " + str(topic) + ":"
		for word in range(0, ldaModel.vocabSize()):
			print " " + str(topics[word][topic])

# def test():
# 	from pyspark.mllib.linalg import Vectors
# 	from pyspark.mllib.linalg import SparseVector
# 	from numpy.testing import assert_almost_equal, assert_equal
# 	from numpy import array
# 	data = [[1, Vectors.dense([0.0, 1.0])],[2, SparseVector(2, {0: 1.0})],]
# 	rdd =  sc.parallelize(data)
# 	model = LDA.train(rdd, k=2, seed=1)
# 	model.vocabSize()
# 	model.describeTopics()
# 	topics = model.topicsMatrix()
# 	topics_expect = array([[0.5,  0.5], [0.5, 0.5]])
# 	print assert_almost_equal(topics, topics_expect, 1)


if __name__ == "__main__":
	import logging, sys
	train_model()
	#load_model()
	#predict()