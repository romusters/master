from plotly.offline import init_notebook_mode, plot

init_notebook_mode()
from plotly.graph_objs import Scatter, Figure, Layout




def plot_sims(seed_id, word):
	import pandas as pd
	data = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_sims_" + str(seed_id) + ".csv", header=None, names=["id", "sims"])
	sims = data.sort("sims", axis=0, ascending=False).sims

	trace = Scatter(x=range(len(sims)), y=sims)

	data = [trace]
	layout = Layout(title="Similarities for word " + word, xaxis=dict(title="n tweets"),
					yaxis=dict(title="Cosine similarity"))
	fig = Figure(data=data, layout=layout)
	plot(fig)

def plot_training_time():
	import pandas as pd
	data = pd.read_csv("/home/cluster/Dropbox/Master/results/optics/training_time.csv")
	x=data.n_tweets
	y = data.training_time
	print y.values.tolist()
	trace = Scatter(x=x, y=y, mode="markers", marker=dict(color="rgb(0,0,0)"))

	data = [trace]
	layout = Layout(title="Insight in to Optics training time", xaxis=dict(title="n tweets"), yaxis=dict(title="training time"))
	fig = Figure(data=data, layout=layout)
	plot(fig)


def plot_cluster_sizes(): #under construction
	import pandas as pd
	data = pd.read_csv("/media/cluster/data1/lambert/cluster_id.csv", header=None)
	freqs = data[0].value_counts()

	trace = Scatter(x=range(len(sims)), y=sims)

	data = [trace]
	layout = Layout(title="Similarities for word " + word, xaxis=dict(title="n tweets"),
					yaxis=dict(title="Cosine similarity"))
	fig = Figure(data=data, layout=layout)
	plot(fig)

def plot_probs():
	import pickle
	fname = "/media/cluster/data1/lambert/probs"
	file = open(fname)
	data = pickle.load(file)
	yes = []
	no = []
	diff = []
	# predictions for training data
	for a,b in data:
		yes.append(b)
		no.append(a)
		diff.append(abs(a) - abs(b))
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler((-3,3))
	diff = scaler.fit_transform(diff)
	print diff

	mult = [a*b for a,b in zip(yes, diff)]
	mult.sort()
	import pandas as pd
	df = pd.DataFrame({"yes": yes, "no": no, "_diff": diff})
	df = df.sort("yes")


	from plotly.offline import init_notebook_mode, plot

	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	# yes.sort()
	t_yes = Scatter(x=range(len(yes)), y=df.yes.values.tolist())
	t_no = Scatter(x=range(len(yes)), y=df.no.values.tolist())
	t_diff = Scatter(x=range(len(yes)), y=df._diff.values.tolist())
	t_mult = Scatter(x=range(len(yes)), y=mult)
	# data = [t_yes, t_no, t_diff]
	data = [t_mult]
	layout = Layout(title="", yaxis=dict(title=""))
	fig = Figure(data=data, layout=layout)
	plot(fig)


def plot_SGD():
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	f1_sgd = [0.74286917727785684, 0.74286917727785684, 0.74539960738080246,0.77816277951035595,  0.80745105723231581, 0.82994632441026861, 0.84891001205219097, 0.90301956605555778]
	acc_sgd = []
	f1_adagrad = [0.80878553458389768, 0.80878553458389768, 0.80951572774643576, 0.82469802830371741, 0.843997039135438, 0.85269550186683296, 0.8604455199174722, 0.8634267176886633 ]
	acc_adagrad = [0.47614348, 0.47614348, 0.4774597, 0.4916091, ]
	x = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 5]
	f1_sgd_trace = Scatter(x=range(len(x)), y=f1_sgd, name="F1 score SGD")
	acc_sgd_trace = Scatter(x=range(len(x)), y=acc_sgd, name="Accuracy SGD")
	f1_adagrad_trace = Scatter(x=range(len(x)), y=f1_adagrad, name="F1 score ADAGRAD")
	acc_adagrad_trace = Scatter(x=range(len(x)), y=acc_adagrad, name="Accuracy ADAGRAD")
	# (precision and recall averaged) of 7 categories where tweets are selected using the category string
	data = [acc_sgd_trace, acc_adagrad_trace, f1_sgd_trace, f1_adagrad_trace]

	layout = Layout(title="F1 score and accuracy compared to Stochastic Gradient Descent and Adagrad learning rate",
					xaxis=dict(title="learning rate", showticklabels=True, ticktext=[str(a) for a in x],
							   tickvals=range(len(x))), yaxis=dict(title="F1-score or accuracy"))


	fig = Figure(data=data, layout=layout)
	plot(fig)

def plot_class_sizes():
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout, Pie
	hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]


	dict = {}
	import pandas as pd
	all = pd.HDFStore("/media/cluster/data1/lambert/datasets/"+ hashtags[0] + ".h5")["data"]
	for hashtag in hashtags[1:]:
		print hashtag
		dict[hashtag] = pd.HDFStore("/media/cluster/data1/lambert/datasets/"+ hashtag + ".h5")["data"]
		all = all.append(dict[hashtag])
		print len(all.index)
	groups = all.groupby("labels").count().id.values.tolist()
	trace = Pie(labels=hashtags, values=groups, textinfo="label+value")
	data = [trace]
	layout = Layout( title="")
	fig = Figure(data=data, layout=layout)
	plot(fig)
	print groups


# plot_probs()
plot_SGD()
# plot_class_sizes()