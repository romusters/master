def hopkins(X, id):
	import numpy as np
	hs = []
	d = len(X[0]) # kolommen
	n = len(X) # rijen
	m = int(0.1 * n)
	import sklearn.preprocessing as pre
	X = pre.normalize(X)
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric="cosine").fit(X)

	from random import sample
	rand_X = sample(range(0, n, 1), m)

	ujd = []
	wjd = []
	for j in range(0, m):
		u_dist, _ = nbrs.kneighbors(pre.normalize(np.random.normal(size=(1, d))).reshape(1, -1), 2, return_distance=True)
		ujd.append(u_dist[0][1])
		w_dist, _ = nbrs.kneighbors(X[rand_X[j]].reshape(1, -1), 2, return_distance=True)
		wjd.append(w_dist[0][1])
	# print ujd, wjd
	H = sum(ujd)/(sum(ujd)+sum(wjd))
	hs.append(H)

	mu_w = np.mean(wjd)
	sigma_w = np.std(wjd)
	mu_u = np.mean(ujd)
	sigma_u = np.std((ujd))

	result = {}
	result["id"] = id
	result["hopkins"] = H
	result["mu_wjd"] = mu_w
	result["mu_ujd"] = mu_u
	result["sigma_w"] = sigma_w
	result["sigma_u"] = sigma_u

	return result

def plot_hopkins(result, id):
	id = result["id"]
	H = result["hopkins"]
	mu_w = result["mu_wjd"]
	mu_u = result["mu_ujd"]
	sigma_w = result["sigma_w"]
	sigma_u = result["sigma_u"]

	# import numpy as np
	# x = np.linspace(-100, 100, 100)
	# import matplotlib.pyplot as plt
	# import matplotlib.mlab as mlab
	# plt.plot(x, mlab.normpdf(x, mu_w, sigma_w), label = "Data")
	# plt.plot(x, mlab.normpdf(x, mu_u, sigma_u), label= "Random data")
	# plt.legend()
	# plt.xlabel("Standard deviation")
	# plt.ylabel("Probability density function")
	# plt.show()

	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	import numpy as np
	import matplotlib.mlab as mlab
	x = np.linspace(-100, 100, 100)
	trace1 = Scatter(x=range(0, len(x)), y=mlab.normpdf(x, mu_w, sigma_w))
	trace2 = Scatter(x=range(0, len(x)), y=mlab.normpdf(x, mu_u, sigma_u))
	data = [trace1, trace2]
	layout = Layout(title='Hopkins value for cluster ' + str(id) + ' is: ' + str(H), xaxis=dict(title="Standard deviation"), yaxis=dict(title="Probability density function"))
	fig = Figure(data=data, layout=layout)
	plot(fig, image="png", auto_open=True, image_filename="hopkins_cluster_" + str(id))

def get_data(id):
	fname = "/home/cluster/w2v_data_cluster_" + str(id) + "_vectors.csv"
	from sklearn import preprocessing
	import pandas as pd
	import numpy as np
	X = pd.read_csv(fname, header=None)
	f = lambda vector: eval(vector.replace("WrappedArray(", "[").replace(")", "]"))
	X = X[2].map(f)
	X = np.array(X.tolist())
	X = preprocessing.scale(X)
	return X, id

# clusters = [8, 16, 111, 140, 189, 190, 231]
# #clusters = [235, 253, 263, 340, 353, 362, 366, 412, 433, 441, 458, 497]
# for cluster in clusters:
# 	print "id ", cluster
# 	print "get data"
# 	data, id = get_data(cluster)
# 	print "get result"
# 	result = hopkins(data, id)
# 	print "plot result"
# 	plot_hopkins(result, id)


def voetbal_moslim():
	import csv_utils
	import numpy as np
	data = csv_utils.rm_wrappedarray()
	print hopkins(np.array(data["vectors"].values.tolist()), 0)

# voetbal_moslim()

# import pandas as pd
# import numpy as np
# data = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")["data"]
# print data.shape
# sample_size = 5
# vectors = map(float, np.array(data.sample(sample_size )[range(70)].values.tolist()).ravel())
# print len(vectors)
# vectors = np.reshape(vectors, (sample_size , data.shape[1]-1))
# print vectors
# hopkins(vectors, 0)