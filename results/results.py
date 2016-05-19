users_coordinates =[[5.88064, 50.4511], [5.99083333, 52.21583333], [5.71666667, 52.85], [5.81989, 51.8404], [6.089741, 52.512205], [6.88611111, 53.10472222], [3.1232688, 50.8029045], [3.1232688, 50.8029045]]

followers_coordinates = [[[4.29522243, 50.8142068], [2.9605401, 51.2226105], [2.74523284, 50.97942068], [-80.1418, 26.1358], [3.73148172, 51.04986943], [-80.19094, 25.76763], [3.28200976, 50.82940704], [4.893, 52.373], [-4.36667, 50.6667], [4.4867902, 52.1570892], [-84.38795932, 33.76225052], [3.13333, 50.95], [-121.92441, 36.55378], [3.2660799, 50.829731], [6.7, 52.9]], [[5.40188366, 52.1706371], [6.07054565, 52.43992125], [5.99653287, 52.24437445], [5.9571941, 52.2014449], [12.44138889, 43.93833333], [5.77638889, 52.875], [5.941599, 52.2126306]], [[5.70583333, 51.54027778], [146.98805556, -43.12166667], [11.573544, 48.148024], [6.3099676, 51.8526374], [5.77638889, 52.875]], [[5.44936111, 51.56765918], [5.70541858, 51.81239046]], [[6.1342219, 52.5228309], [6.46667, 52.3667], [5.36253664, 52.15651534], [6.089741, 52.512205]], [[7.0817089, 53.0823193]], [[4.5, 51.9167], [77.545513, 12.9251151], [5.8705655, 52.8101945]], [[3.56367758, 51.18211949]]]

follower_length = [672, 647, 265, 178, 151, 140, 95, 89]

coordinates_found = [15, 7, 5, 2, 4, 1, 3, 1]

a = sum(follower_length)/len(follower_length)
b = sum(coordinates_found)/len(coordinates_found)

print a
print b
print a/b

res = 0
for i in range(len(follower_length)):
	res += float(coordinates_found[i])/follower_length[i]

res /= len(follower_length)
print res



import validateGeo

print validateGeo.center_geolocation(followers_coordinates[0])

import sklearn.cluster
import numpy as np
X = followers_coordinates[0]
print X

db = sklearn.cluster.DBSCAN().fit(followers_coordinates[0])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print db.components_
print db.get_params()

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print n_clusters_

# Plot result
import matplotlib.pyplot as plt

plt.title('Estimated number of clusters: %d' % n_clusters_)
for p in X:
	plt.scatter(p[0], p[1])
plt.show()