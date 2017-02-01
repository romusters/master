from plotly.offline import init_notebook_mode, plot
init_notebook_mode()
from plotly.graph_objs import Scatter, Figure, Layout, Pie



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

def plot_class_sizes():
	hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]
	import balance_categories
	data = balance_categories.load_balanced_data()
	groups = data.groupby("labels").count().id.values.tolist()
	trace = Pie(labels=hashtags, values=groups, textinfo="label+value")
	data = [trace]
	layout = Layout( title="")
	fig = Figure(data=data, layout=layout)
	plot(fig)
	print groups



import pandas as pd
def plot_w2v_loss():
	data = pd.read_csv("/home/cluster/Dropbox/Master/results/model_lambert_w2v_loss.csv")
	wordcount = data["wordcount"].values.tolist()
	eta = data["eta"].values.tolist()
	# print type(data["wordcount"][1])
	# data["wordcount", "eta"] = data.apply(lambda row: eval(row))
	trace = Scatter(x=wordcount, y=eta)
	layout = Layout(title="N words versus learning rate", xaxis=dict(title="Wordcount"), yaxis=dict(title="Learning rate"))
	fig = Figure(data=[trace], layout=layout)
	plot(fig)



def plot_loss(classifier):
	from dataset import variables
	traces = []
	key = "loss"
	for i, var in enumerate(variables):
		data = pd.read_csv("/media/cluster/data1/lambert/results/"+ classifier + "_result_" + str(var))
		traces.append(Scatter(x=range(len(data[key])), y=data[key], name=var,  line = dict(width = 1,color=colors[i]), marker= dict(opacity= 0.01),opacity= 0.3))
		y = np.convolve(data[key], np.ones((50,)) / 50, mode="valid")
		traces.append(Scatter(x=range(len(data[key])), y=y, name=var, showlegend=False, line=dict(width=1, color=colors[i])))
	layout = Layout(title="Loss versus iterations for different "+ classifier.upper() + " learning rates", xaxis=dict(title="Iterations"), yaxis=dict(title="Loss (cross entropy)"))
	fig = Figure(data=traces, layout=layout)
	plot(fig)


def plot_train_acc(classifier):
	from dataset import variables
	traces = []
	key = "train_acc"
	for i, var in enumerate(variables):
		data = pd.read_csv("/media/cluster/data1/lambert/results/" + classifier + "_result_" + str(var))
		traces.append(Scatter(x=range(len(data[key])), y=data[key], name=var,  line = dict(width = 1,color=colors[i]), marker= dict(opacity= 0.01),opacity= 0.3))
		y = np.convolve(data[key], np.ones((50,)) / 50, mode="valid")
		traces.append(Scatter(x=range(len(data[key])), y=y, name=var, showlegend=False, line=dict(width=1, color=colors[i])))
	layout = Layout(title="Training accuracy versus iterations for different "+ classifier.upper() + " learning rates", xaxis=dict(title="Iterations"), yaxis=dict(title="Training accuracy"))
	fig = Figure(data=traces, layout=layout)
	plot(fig)

def plot_test_acc(classifier):
	from dataset import variables
	traces = []
	key = "test_acc"
	for i, var in enumerate(variables):
		data = pd.read_csv("/media/cluster/data1/lambert/results/" + classifier + "_result_" + str(var))
		traces.append(Scatter(x=range(len(data[key])), y=data[key], name=var,  line = dict(width = 1,color=colors[i]), marker= dict(opacity= 0.01),opacity= 0.3))
		y = np.convolve(data[key], np.ones((50,)) / 50, mode="valid")
		traces.append(Scatter(x=range(len(data[key])), y=y, name=var, showlegend=False,line=dict(width=1, color=colors[i])))
	layout = Layout(title="Test accuracy versus iterations for different "+ classifier.upper() + " learning rates", xaxis=dict(title="Iterations"), yaxis=dict(title="Test accuracy"))
	fig = Figure(data=traces, layout=layout)
	plot(fig)

def plot_f1(classifier):
	from dataset import variables
	traces = []
	key = "f1"
	for i, var in enumerate(variables):
		data = pd.read_csv("/media/cluster/data1/lambert/results/" + classifier + "_result_" + str(var))
		traces.append(Scatter(x=range(len(data[key])), y=data[key], name=var,  line = dict(width = 1,color=colors[i]), marker= dict(opacity= 0.01),opacity= 0.3))
		y = np.convolve(data[key], np.ones((50,)) / 50, mode="valid")
		traces.append(Scatter(x=range(len(data[key])), y=y, name=var, showlegend=False, line=dict(width=1, color=colors[i])))
	layout = Layout(title="F1 score versus iterations for different "+ classifier.upper() + " learning rates",
					xaxis=dict(title="Iterations"), yaxis=dict(title="F1 score"))
	fig = Figure(data=traces, layout=layout)
	plot(fig)


def plot_train_test():

	trace_train = Scatter(x=range(len(train)), y=train)
	trace_test = Scatter(x=range(len(test)), y=test)
	fig = Figure(data=[trace_train, trace_test])
	plot(fig)
# plot_train_test()


def plot_probs():
	data = pd.read_csv("/media/cluster/data1/lambert/results/voetbal_moslim.csv", header=None, low_memory=False)
	data = data.dropna()
	data[2] = data[2].apply(lambda x: float(x))
	data = data.sort(columns=[2])
	print len(data.index)
	old_data  = data
	data = data[data[3].str.contains("voetbal")]
	print len(data.index)
	probs = data[2].dropna()
	trace = Scatter(x=range(len(probs)), y=probs)
	trace2 = Scatter(x=range(len(old_data[2].dropna())), y=old_data[2].dropna())
	fig = Figure(data=[trace, trace2])
	plot(fig)
plot_probs()
# plot_probs()
# plot_SGD()
# plot_class_sizes()

# plot_w2v_loss()
# import sys
# sys.exit(0)
#
#
#
# import time
# import numpy as np
# colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "purple", "brown", "lime", "steelblue", "chocolate", "orange"]
# classifiers = ["ada", "sgd"]
# for c in classifiers:
# 	plot_loss(c)
# 	time.sleep(2)
# 	plot_train_acc(c)
# 	time.sleep(2)
# 	plot_test_acc(c)
# 	time.sleep(2)
# 	plot_f1(c)
# 	time.sleep(2)
# plot_train_acc("sgd")