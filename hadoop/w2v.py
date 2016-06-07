from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF
import logging, sys
import filter

# The command
# spark-submit --master yarn --executor-memory 50g --driver-memory 50g --deploy-mode cluster --num-executors 1000 master/hadoop/w2v.py
#--driver-memory 50g --executor-memory 6g ----num-executors 400

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Small file
#loc = '/user/rmusters/text/2010/12/20101231-23.out.gz.txt'

# Larger file
loc = '/user/rmusters/text/2015/01/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("driver-memory", "50g")\
	.set("executor-memoery", "6g")\
	.set("num-executors", "400"))

sc = SparkContext(appName='Word2Vec', conf=conf)

# Threshold to limit words which occur less than the threshold
threshold = 10
text_file = sc.textFile(loc)

data = text_file.map(lambda line: filter.filter(line))

counts = data.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b) \
			.filter(lambda pair: pair[1] >= threshold) \
			.sortBy(lambda x:x[1], ascending=True)
counts.cache()

vocab_size = counts.count()
print "Vocabulary size is: ", vocab_size

inp = data.map(lambda line: line.split(" "))
inp.cache()


max_int_size = 268435455
vector_size = max_int_size / vocab_size
print "Vector size is: ", vector_size

word2vec = Word2Vec()
word2vec.setMinCount(40)
word2vec.setVectorSize(vector_size/100)
model = word2vec.fit(inp)

model.save(sc, '/user/rmusters/2015modelfull2')

#model =  word2vec.load(sc, '/user/rmusters/pymodel.bin')
