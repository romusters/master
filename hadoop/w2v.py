from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF
import logging, sys
import filter

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


loc = '/user/rmusters/text/2010/12/20101231-23.out.gz.txt'
#loc = '/user/rmusters/text/2015/01/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='Word2Vec', conf=conf)


text_file = sc.textFile(loc)
counts = text_file.map(lambda line: filter.filter(line)) \
             .flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)\
			 .sortBy(lambda x:x[1], ascending=True)
unique_words = counts.count()
print "Unique words are: ", unique_words

inp = sc.textFile(loc).map(lambda line: filter.filter(line).split(" "))\

# inp.cache()

# inp = sc.textFile(loc).map(lambda row: row.split(" "))
# hashingTF = HashingTF()
# tf = hashingTF.transform(inp)
# print tf.count()

vocab_size = unique_words
print "Vocabulary size is: ", vocab_size

max_int_size = 268435455
# vector_size = max_int_size / tf.count()
# print vector_size

word2vec = Word2Vec()
# word2vec.setMinCount(500)

vector_size = max_int_size / vocab_size
print "Vector size is: ", vector_size

word2vec.setVectorSize(vector_size)
model = word2vec.fit(inp)

model.save(sc, '/user/rmusters/pymodel_filtered6')

#model =  word2vec.load(sc, '/user/rmusters/pymodel.bin')

#print model.getVectors()


