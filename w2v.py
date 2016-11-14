import pandas as pd
import numpy as np
import math

# remove WrappedArray from embeddings from hadoop model
model_name = "/media/cluster/data1/vectors_threshold20_2015model56.csv.clean"
data_name = "/media/cluster/data1/data_sample.csv"
# data = pd.read_csv(data_name + ".clean", header=None, iterator=True, chunksize=1)
# chunk = data.get_chunk()
# print chunk

def clean_data_column(data_name, header=None):
	data = pd.read_csv(data_name, header=header)
	f = lambda vector: eval(vector.replace("WrappedArray(", "[").replace(")", "]"))
	data[1] = data[1].map(f)
	data.to_csv(data_name + ".clean")
	return data


def clean_data():
	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=1)
	idx = 0
	chunk = data.get_chunk()
	# chunk.rename(columns={0: 'words', 1: 'tokens', 2: 'vector', 3: 'cluster'}, inplace=True)
	while chunk is not None:
		if idx % 1000 == 0:
			print idx
		try:
			f = lambda vector: eval(vector.replace("WrappedArray(", "[").replace(")", "]"))
			# chunk = chunk[chunk.vector != 'null']
			tmp = chunk[2] != 'null'
			if tmp[0]:
				clean_vectors = chunk[2].map(f)
				clean_data = chunk
				clean_data[2] = clean_vectors
				# clean_data = clean_data.drop("cluster", 1)
				clean_data.to_csv(data_name + ".clean", index=None, mode="a", header=False)
			chunk = data.get_chunk()
			# chunk.rename(columns={0: 'words', 1: 'tokens', 2: 'vector', 3: 'cluster'}, inplace=True)
			idx += 1
		except StopIteration:
			break
# clean_data()

	#
def cosine_similarity(v1,v2):
	# compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
	sumxx, sumxy, sumyy = 0, 0, 0
	for i in range(len(v1)):
		x = v1[i]; y = v2[i]
		sumxx += x*x
		sumyy += y*y
		sumxy += x*y
	return sumxy/math.sqrt(sumxx*sumyy)
	#return sumxy/math.sqrt(sumyy)

chunk_size = 1000
def predict():
	data_name = "/media/cluster/data1/data_sample.csv.clean"

	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=chunk_size)
	chunk = data.get_chunk()
	print chunk.keys()
	vector_seed = eval(chunk[2][0])

	idx = 0
	sims_name = data_name + ".sims"
	while chunk is not None:
		print idx

		try:
			sims = chunk[2].apply(lambda x: cosine_similarity(eval(x), vector_seed))
			chunk[4] = sims
			chunk.to_csv(sims_name, index=None, mode="a", header=None)
			chunk = data.get_chunk()
		except StopIteration:
			break
		idx += 1


def sort_predicted():
	data = pd.read_csv(data_name + ".clean.sims", header=None, usecols=[4], nrows=10000)
	data = data.sort_values([4], ascending=False)
	return data

def plot_sort_predicted(data):
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	trace = Scatter(x=range(data.count()), y=np.array(data.values.tolist()).ravel(), mode="lines+markers")
	data = [trace]
	layout = Layout(title="", xaxis=dict(title=""),
					yaxis=dict(title=""))
	fig = Figure(data=data, layout=layout)
	plot(fig)
	print data


def get_nearest():
	sorted = sort_predicted()
	result = sorted[sorted[4] > 0.5].index.values
	print result
	return result

def get_data_by_index(data, indices):
	print data.loc[indices]



def test():
	data_name = "/media/cluster/data1/test.csv"
	f = open(data_name + ".clean", "a")
	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=1)
	chunk = data.get_chunk().drop(3, 1)
	while chunk is not None:
		try:
			tmp = ";".join(chunk.values.tolist()[0])
			print tmp
			f.write(tmp + "\n")
			chunk = data.get_chunk().drop(3, 1)
		except StopIteration:
			break
	f.close()
# test()

def test2():
	data_name = "/media/cluster/data1/test.csv"
	f = open(data_name + "2.clean", "a")
	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=2)
	chunk = data.get_chunk().drop(3, 1)
	while chunk is not None:
		try:
			tmp = ";".join(chunk.values.tolist()[0])
			print tmp
			f.write(tmp + "\n")
			chunk = data.get_chunk().drop(3, 1)
		except StopIteration:
			break
	f.close()
# test2()

# X = pd.read_csv(fname, header=None)
# f = lambda vector: eval(vector.replace("WrappedArray(", "[").replace(")", "]"))
# clean_vectors = X[1].map(f)
#
# X[1] = clean_vectors
# X.columns = ["words", "vectors"]
# print X
# X.to_csv(fname + ".clean", index=None)


def AL():
jan = True
if jan:
	emb_name = "/media/cluster/data1/vectors_jan_threshold20_2015model99.csv.clean"
	embs = pd.read_csv(emb_name)
	dim = len(embs[1][0])
else:
	emb_name = "/media/cluster/data1/vectors_threshold20_2015model56.csv.clean"
	embs = pd.read_csv(emb_name)
	dim = len(embs["vectors"][0])

print embs.keys()


data_name = "/media/cluster/data1/data_sample.csv"
data = pd.read_csv(data_name + ".clean.sims", header=None, nrows=10000)
sorted = sort_predicted()
yes = data.loc[sorted[sorted[4] > 0.5].index.values.tolist()].sample(20)
yes[2] = yes[2].apply(lambda x: eval(x))
yes[5] = pd.DataFrame(data=[dim*[[0,1]]]).T

no = data.loc[sorted[(sorted[4] < 0.5) & (sorted[4] > 0.0)].index.values.tolist()].sample(20)
no[2] = no[2].apply(lambda x: eval(x))
no[5] = pd.DataFrame(data=[dim*[[1,0]]]).T
batch = yes.append(no)

import tensorflow as tf
sess = tf.InteractiveSession()

n_classes = 2
x = tf.placeholder(tf.float32, shape=[None, dim])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
W = tf.Variable(tf.zeros([dim, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))
sess.run(tf.initialize_all_variables())

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

train_step.run(feed_dict={x: batch[2].values[0], y_: batch[5].values[0]})


def main():
	data = pd.read_csv(fname)

	A = np.array(eval(data[data["words"] == "moslim"].vectors.values.tolist()[0]))
	sims = data.vectors.apply(lambda x: cosine_similarity(eval(x), A))


	sims.sort(ascending=False)
	els = sims.head(10).keys()

	print els

	for el in els:
		print data.words[el]

	# B = np.array(eval(data[data["words"] == "drie"].vectors.values.tolist()[0]))
	# print A
	# print B
	# from scipy import spatial
	#
	# result = 1 - spatial.distance.cosine(A, B)
	# print result

#
# print cosine_similarity(A, B)

