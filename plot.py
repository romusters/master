def plot_sims(seed_id, word):
	import pandas as pd
	data = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_sims_" + str(seed_id) + ".csv", header=None, names=["id", "sims"])
	sims = data.sort("sims", axis=0, ascending=False).sims
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
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
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	trace = Scatter(x=x, y=y, mode="markers", marker=dict(color="rgb(0,0,0)"))

	data = [trace]
	layout = Layout(title="Insight in to Optics training time", xaxis=dict(title="n tweets"), yaxis=dict(title="training time"))
	fig = Figure(data=data, layout=layout)
	plot(fig)

# plot_sims()
