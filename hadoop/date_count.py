from pyspark import SparkContext, SparkConf
from datetime import datetime
from operator import attrgetter, add

#hadoop -f -rm -r /user/rmusters/date_
#spark-submit --driver-memory 8g --executor-memory 8g --master yarn  --deploy-mode cluster --num-executors 800 master/hadoop/date_count.py
#hdfs -dfs -getmerge /user/rmusters/date_

f = "%a %b %d %X +0000 %Y"
loc = '/user/rmusters/dates.txt'
# loc = "/home/robert/Downloads/dates.txt"

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='date_count', conf=conf)


rdd = sc.textFile(loc)



def check(date):
	date_list = date.split()
	if len(date_list) == 6:
		return date
	else:
		return 'Thu Dec 16 18:53:32 +0000 1900'

def parse_line(d):
	date_list = d.split()
	date = date_list[:4]
	date.append(date_list[5])
	date = ' '.join(date)
	return datetime.datetime.strptime(date, f)

	# .map(parse_line)\
counts = rdd.map(lambda x: datetime.strptime(check(x), f))\
	.map(attrgetter('year', 'month', 'day'))\
	.filter(lambda (y, m, d): y == 2014)\
	.map(lambda x: (str(x), 1)).reduceByKey(add)
	# .countByValue()

counts.saveAsTextFile('/user/rmusters/dates_2014')