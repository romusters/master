from pyspark import SparkContext, SparkConf
import datetime
from operator import attrgetter


f = "%a %b %d %X %Y"
loc = '/user/rmusters/dates.txt'
# loc = "/home/robert/Downloads/dates.txt"

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='date_count', conf=conf)


rdd = sc.textFile(loc)

def parse_line(d):

	date_list = d.split()
	date = date_list[:4]
	date.append(date_list[5])
	date = ' '.join(date)
	return datetime.datetime.strptime(date, f)

	# .map(parse_line)\
counts = rdd.map(lambda x: datetime.strptime(x, f))\
	.map(attrgetter('year', 'month', 'day'))\
	.filter(lambda (y, m, d): y == 2015)\
	.countByValue()

counts.saveAsTextFile('/user/rmusters/date_freqs')