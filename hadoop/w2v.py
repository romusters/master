from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF

loc = '/user/rmusters/text/2010/12/20101231-23.out.gz.txt'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='Word2Vec', conf=conf)


inp = sc.textFile(loc).map(lambda row: row.split(" "))
hashingTF = HashingTF()
tf = hashingTF.transform(inp)
print tf.count()



max_int_size = 268435455
vector_size = max_int_size / tf.count()
print vector_size

word2vec = Word2Vec()
word2vec.setMinCount(500)

word2vec.setVectorSize(vector_size)
model = word2vec.fit(inp)

# model.save(sc, '/user/rmusters/pymodel.bin')

# model =  Word2VecModel.load(sc, '/user/rmusters/pymodel.bin')

# print model.getVectors()


