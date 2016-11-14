
import pandas as pd
data = pd.read_csv("/home/cluster/Dropbox/Master/results/optics/training_time.csv")
x=data.n_tweets
y = data.training_time
print y.values.tolist()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
trace = Scatter(x=x, y=y, mode="markers", marker=dict(color="rgb(0,0,0)"))

data = [trace]
layout = Layout(title="Insight in to Optics trainging time", xaxis=dict(title="n tweets"), yaxis=dict(title="training time"))
fig = Figure(data=data, layout=layout)
plot(fig)