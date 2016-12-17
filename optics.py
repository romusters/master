def read_data(id):
	import numpy as np

	fname = "/home/cluster/w2v_data_cluster_" + str(id) + "_vectors.csv"
	#fname = "/home/cluster/w2v_vectors_sample.csv"
	import pandas
	data = pandas.read_csv(fname, header=None)
	f = lambda vector: eval(vector.replace("WrappedArray(", "[").replace(")", "]"))
	data = data[2].map(f)
	el = data[0]

	print len(data)
	print len(el)

	return np.array(data.tolist())


def optics_alg(x, k, distMethod='cosine'): #was euclidean
	import time
	tic = time.clock()
	import numpy as N
	import pylab as P
	import hcluster as H
	if len(x.shape) > 1:
		m, n = x.shape
	else:
		m = x.shape[0]
		n == 1

	try:
		# D = H.squareform(H.pdist(x, distMethod))
		from scipy.spatial.distance import pdist
		D = H.squareform(pdist(x, distMethod))
		distOK = True
	except:
		print "squareform or pdist error"
		distOK = False

	CD = N.zeros(m)
	RD = N.ones(m) * 1E10

	for i in xrange(m):
		# again you can use the euclid function if you don't want hcluster
		#        d = euclid(x[i],x)
		#        d.sort()
		#        CD[i] = d[k]

		tempInd = D[i].argsort()
		tempD = D[i][tempInd]
		#        tempD.sort() #we don't use this function as it changes the reference
		CD[i] = tempD[k]  # **2

	order = []
	seeds = N.arange(m, dtype=N.int)

	ind = 0
	while len(seeds) != 1:
		#    for seed in seeds:
		ob = seeds[ind]
		seedInd = N.where(seeds != ob)
		seeds = seeds[seedInd]

		order.append(ob)
		tempX = N.ones(len(seeds)) * CD[ob]
		tempD = D[ob][seeds]  # [seeds]
		# you can use this function if you don't want to use hcluster
		# tempD = euclid(x[ob],x[seeds])

		temp = N.column_stack((tempX, tempD))
		mm = N.max(temp, axis=1)
		ii = N.where(RD[seeds] > mm)[0]
		RD[seeds[ii]] = mm[ii]
		ind = N.argmin(RD[seeds])
	toc = time.clock()
	res = toc - tic
	print "Compute time is: ", res
	order.append(seeds[0])
	RD[0] = 0  # we set this point to 0 as it does not get overwritten
	return RD, CD, order

def plot_optics(ordered_reachabilities, id):
	from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
	init_notebook_mode()
	import plotly.plotly as py
	from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
	from plotly.graph_objs import Bar, Figure, Layout
	trace = Bar(x=range(0, len(ordered_reachabilities)), y=ordered_reachabilities)
	data = [trace]
	layout = Layout(title='Optics algoritme for cluster: ' + str(id), xaxis=dict(title="index"), yaxis=dict(title="Reachability"))
	fig = Figure(data=data, layout=layout)
	plot(fig,  image="png", auto_open=True, image_filename="optics_cluster_" + str(id))

def main():
	clusters = [8, 16, 111, 140, 189, 190, 231, 235, 253, 340,263, 353, 362, 366, 412, 433, 441, 458, 497]
	#clusters = [412]
	for cluster in clusters:
		data = read_data(cluster)
		fname = "/home/cluster/w2v_data_cluster_" + str(cluster) + "_vectors.csv"
		import pandas
		tweets = pandas.read_csv(fname, header=None)[2]
		print tweets.count()
		RD, CD, order = optics_alg(data, 10)
		# idxs = order[29:443]
		# # idxs = order[1099:1112]
		# tweets = tweets[idxs]
		# for tweet in tweets:
		# 	print tweet
		# ordered_reachabilities = RD[order]
		# plot_optics(ordered_reachabilities, cluster)

import pandas as pd
import numpy as np
data = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")["data"]
vectors = np.array(data.sample(20000)[range(70)].values.tolist())
RD, CD, order = optics_alg(vectors, 10)
ordered_reachabilities = RD[order]
ordered_reachabilities = [ o for o in ordered_reachabilities if o <= 1]
plot_optics(ordered_reachabilities, 11)