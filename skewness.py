def get_skewness():
	import pandas as pd
	dim = 70
	fname = None
	data = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")["data"][range(70)]
	skewness = data.skew().values.tolist()
	print skewness
	return skewness

def plot_skewness(data):
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Histogram, Scatter, Figure, Layout
	dim = len(data)
	data = get_skewness()
	# trace = Scatter(x=data, y=range(dim), mode="markers", marker=dict(color="rgb(0,0,0)"))
	trace = Histogram(x=data, xbins=dict(size=0.2))

	data = [trace]
	layout = Layout(title="Insight into skewness", xaxis=dict(title="Feature"),
					yaxis=dict(title="Skewness"))
	fig = Figure(data=data, layout=layout)
	plot(fig)

skewness = get_skewness()
plot_skewness(skewness)


# def get_first_attr():
# 	import numpy as np
# 	import sys
# 	import csv
# 	path = '/home/cluster/data/lda_data_cluster.csv'
#
#
# 	idx = 0
# 	dict = {}
# 	import scipy.stats
# 	total_skewness = []
# 	for i in range(0, 10774, 1):
# 		f = open(path, "r")
# 		data = csv.reader(f)
# 		vals = []
# 		print i
# 		for row in data:
# 			vals.append(eval(row[2])[i])
# 		skew = scipy.stats.skew(vals)
# 		print skew
# 		total_skewness.append(skew)
# 	print total_skewness
