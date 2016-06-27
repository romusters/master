from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
import logging, sys


#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --master yarn --deploy-mode cluster master/hadoop/lda.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Larger file
loc = '/user/rmusters/text/2015/*/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("driver-memory", "6g")\
	.set("executor-memory", "6g")\
	.set("num-executors", "200"))

sc = SparkContext(appName='lda', conf=conf)


from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

model_vectors = sqlContext.read.parquet('/user/rmusters/2015model99/data')

rdd_words = model_vectors.map(lambda line: line[0])
words = rdd_words.collect() #15919

def load_data(path):
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet(path)
	return data

path = 'hdfs:///user/rmusters/data'
data = load_data(path)
data = data.sample(False, 0.01)
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
	return v_dict

from pyspark.mllib.linalg import SparseVector
size = len(words)
data = data.map(lambda (text, filtered_text, id): (text, filtered_text, SparseVector(size, bow(filtered_text)), id))

corpus = data.map(lambda (text, filtered_text, vector, id): [id, vector])
ldaModel = LDA.train(corpus, k=100)
logger.info(ldaModel.vocabSize())
logger.info(ldaModel.describeTopics())
ldaModel.save(sc, 'hdfs:///user/rmusters/ldaModel')