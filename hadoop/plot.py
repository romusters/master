#https://plot.ly/python/offline/
def plot_cluster_freqs():
	import matplotlib.pyplot as plt
	import numpy as np
	import csv
	path = "/home/cluster/Dropbox/Master/results/cluster_freqs"

	f = open(path, "r")
	data = csv.reader(f)

	range = []
	w2v = []
	lda = []

	for row in data:
		print row

		range.append(int(row[0]))
		w2v.append(int(row[1]))
		lda.append(int(row[2]))
		print row
	range = np.arange(len(range))
	plt.bar(range-0.2, lda, 0.2, color='r')
	plt.bar(range+0.2, w2v, 0.2, color='b')
	plt.show()


def plot_skewness():
	fname = "/home/cluster/Dropbox/Master/results/skewness"
	f = open(fname, 'r')
	data = []
	while f:
		index = f.readline()

		val = f.readline()
		# print val
		data.append(float(val))
		if int(index) == 499:
			break
	# import plotly.plotly
	# plotly.offline.plot({
	# 	"data": [
	# 		plotly.graph_objs.Histogram(x=data, y=range(0, len(data)))
	# 	],
	# 	"layout": [
	# 		title = "Titel",
	# 	]
	# })
	from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

	init_notebook_mode()
	import plotly.plotly as py
	from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
	from plotly.graph_objs import Histogram, Scatter, Figure, Layout
	trace = Histogram(x=data, y=range(0, len(data)))
	data = [trace]
	layout = Layout(title='Skewness for every W2V feature', xaxis=dict(title="skewness"),yaxis=dict(title="frequency"))
	fig = Figure(data=data, layout=layout)
	plot(fig)
	py.image.save_as(fig, 'my_plot.png')
	from IPython.display import Image
	Image(filename='my_plot.png')
	print data



def plot_pies():
	fname = "/home/cluster/Dropbox/Master/results/cluster_freqs_w2v_lda"
	f = open(fname, 'r')
	clusters = []
	values = []
	data = f.readlines()
	for i,e in enumerate(data):
		if i%2 == 0:
			clusters.append(eval(e))
		else:
			values.append(eval(e))
	print clusters
	print values


	pie_values = []
	labels = []
	for idx,l in enumerate(values):
		print type(idx)
		l_set = set(l)
		tmp_labels = []
		tmp_pie_values = []
		tmp_pie_values.append(list(l_set))
		for i in l_set:
			tmp_labels.append((l.count(i)/2.0)*10)
		pie_values.append(tmp_pie_values)
		labels.append(tmp_labels)


		from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
		from plotly.graph_objs import *
		init_notebook_mode()
		print pie_values[idx][0]
		trace = Pie(labels=pie_values[idx][0], values=labels[idx])
		data = [trace]

		layout = Layout(title='Cluster id ' + str(clusters[idx]))
		fig = dict(data=data, layout=layout)
		plot(fig)
		import time
		time.sleep(2)




# plot_pies()
plot_skewness()
# plot_cluster_freqs()

# pyspark --master local --deploy-mode client --packages com.databricks:spark-csv_2.10:1.4.0
# (df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferschema", "true").option("mode", "DROPMALFORMED").load("/media/cluster/dataThesis/lda_data_cluster.csv"))