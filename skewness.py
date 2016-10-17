def get_first_attr():
	import numpy as np
	import sys
	import csv
	path = '/home/cluster/data/lda_data_cluster.csv'


	idx = 0
	dict = {}
	import scipy.stats
	total_skewness = []
	for i in range(0, 10774, 1):
		f = open(path, "r")
		data = csv.reader(f)
		vals = []
		print i
		for row in data:
			vals.append(eval(row[2])[i])
		skew = scipy.stats.skew(vals)
		print skew
		total_skewness.append(skew)
	print total_skewness
