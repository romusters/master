import pandas as pd
data = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")["data"][range(70)]
data = data.dropna()
data = data.sample(100000)


inertias = []
from sklearn.cluster import KMeans
for i in range(400, 500, 10):
	print i
	kmeans = KMeans(n_clusters=i, random_state=0, n_jobs=-1).fit(data.values.tolist())
	print kmeans.inertia_
	inertias.append(kmeans.inertia_)

print inertias