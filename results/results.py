# users_coordinates =[[5.88064, 50.4511], [5.99083333, 52.21583333], [5.71666667, 52.85], [5.81989, 51.8404], [6.089741, 52.512205], [6.88611111, 53.10472222], [3.1232688, 50.8029045], [3.1232688, 50.8029045]]
#
# followers_coordinates = [[[4.29522243, 50.8142068], [2.9605401, 51.2226105], [2.74523284, 50.97942068], [-80.1418, 26.1358], [3.73148172, 51.04986943], [-80.19094, 25.76763], [3.28200976, 50.82940704], [4.893, 52.373], [-4.36667, 50.6667], [4.4867902, 52.1570892], [-84.38795932, 33.76225052], [3.13333, 50.95], [-121.92441, 36.55378], [3.2660799, 50.829731], [6.7, 52.9]], [[5.40188366, 52.1706371], [6.07054565, 52.43992125], [5.99653287, 52.24437445], [5.9571941, 52.2014449], [12.44138889, 43.93833333], [5.77638889, 52.875], [5.941599, 52.2126306]], [[5.70583333, 51.54027778], [146.98805556, -43.12166667], [11.573544, 48.148024], [6.3099676, 51.8526374], [5.77638889, 52.875]], [[5.44936111, 51.56765918], [5.70541858, 51.81239046]], [[6.1342219, 52.5228309], [6.46667, 52.3667], [5.36253664, 52.15651534], [6.089741, 52.512205]], [[7.0817089, 53.0823193]], [[4.5, 51.9167], [77.545513, 12.9251151], [5.8705655, 52.8101945]], [[3.56367758, 51.18211949]]]
#
# follower_length = [672, 647, 265, 178, 151, 140, 95, 89]
#
# coordinates_found = [15, 7, 5, 2, 4, 1, 3, 1]

# determine ratio
def ratio(follower_length, n_coordinates_found):
	a = sum(follower_length)/len(follower_length)
	b = sum(n_coordinates_found)/len(n_coordinates_found)

	print "Average number of followers is : ", a
	print "Average number of coordinates found is: ", b
	print "Average number of coordinates found per follower is: ", float(b)/a

	print "Percentage of no coordinates found is", float(n_coordinates_found.count(0))/len(n_coordinates_found)
	res = 0
	for i in range(len(follower_length)):
		res += float(n_coordinates_found[i])/follower_length[i]


	res /= len(follower_length)
	print res


def test(followers_coordinates, users_coordinates):
	import validateGeo
	import utils
	print validateGeo.center_geolocation(followers_coordinates[0])

	import sklearn.cluster
	import numpy as np

	results = []
	eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	min_samples_range = range(1, 6, 1)
	for eps_ in eps_range:
		for min_samples_ in min_samples_range:
			dbs = []
			empties = 0
			for i, X in enumerate(followers_coordinates):
				if len(X) == 0:
					continue
				#print "True coordinate is: ", users_coordinates[i]
				db = sklearn.cluster.DBSCAN(eps=eps_, min_samples=min_samples_).fit(X) #eps = 0.5 min_samples=3
				core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
				core_samples_mask[db.core_sample_indices_] = True
				labels = db.labels_

				#possibly component of largest cluster
				#print "predicted cluster is: ", db.components_.tolist()
				if len(db.components_.tolist()) is 0:
					empties = empties +1
					continue
				#print "predicted center is: COM", validateGeo.center_geolocation(X)
				#print "predicted center is: DB", validateGeo.center_geolocation(db.components_.tolist())
				dbs.append(int(abs(utils.distance(users_coordinates[i], validateGeo.center_geolocation(db.components_.tolist())))))

				#print "error is: COM: ", int(abs(utils.distance(users_coordinates[i], validateGeo.center_geolocation(X))))
				#print "error is: DB: ",	int(abs(utils.distance(users_coordinates[i], validateGeo.center_geolocation(db.components_.tolist()))))

				# Number of clusters in labels, ignoring noise if present.
				n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
				#print "number of clusters is: ", n_clusters_
			if eps_ == 0.1:
				if min_samples_ == 5:
					print "DBSCAN DISTANCES: ", dbs, sum(dbs)/len(followers_coordinates)-(empties)
			results.append([eps_, min_samples_, sum(dbs)/len(followers_coordinates)-(empties), empties])
			#results.append(sum(dbs)/(len(followers_coordinates)-empties))
		# Plot result
	normal_distances = []
	for i, X in enumerate(followers_coordinates):
		if len(X) == 0:
			continue
		normal_distances.append(int(abs(utils.distance(users_coordinates[i], validateGeo.center_geolocation(X)))))
	print normal_distances #[2393, 2944, 15993, 21772, 31569, 36331, 56356, 64402, 82800, 100519, 110708, 147478, 157583, 235925, 256514, 405028, 437401, 636586, 1458614, 2575768, 3016883, 3894907, 5428058, 5533571, 5719402, 5820953, 6195934, 6624939, 6668912, 6687665, 6721412, 6758942, 6811755]
	normal_distances.sort()
	import matplotlib.pyplot as plt
	#
	# plt.title('Estimated number of clusters: %d' % n_clusters_)
	# for p in X:
	# 	plt.scatter(p[0], p[1])
	# plt.show()

	scores = []
	empties = []
	for e in results:
		empties.append(e[3])
		scores.append(e[2])
		if e[2] == 137493:
			print e
		#[0.1, 5, 4608, 25]
		#[0.2, 5, 4520, 25]
	#distances for dbscan [3655, 3727, 111539, 111575, 111606, 111691, 111705, 218346, 218429, 218465, 218470, 218490, 218612, 219833, 219871, 219953, 220169, 220169, 220255, 220483, 220593, 220972, 343756, 345028, 345358, 348025, 348118, 348446, 348525, 350793, 351421, 351465, 351556, 351562, 352276, 352566, 352615, 352881, 457393, 457393, 1560755, 1560755, 1560755, 1560755, 1560755, 1560755, 1560755, 1560755, 1560755, 1560755]
	# #scores.sort()
	t = scores
	t.sort()


	print scores #smallest score is 4520
	print empties
	# scores  = results
	print len(scores)
	from matplotlib.colors import Normalize
	class MidpointNormalize(Normalize):
		def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
			self.midpoint = midpoint
			Normalize.__init__(self, vmin, vmax, clip)

		def __call__(self, value, clip=None):
			x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
			return np.ma.masked_array(np.interp(value, x, y))

	scores = np.array(scores).reshape(5, 10)
	plt.figure(figsize=(8, 6))
	plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
			   norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
	plt.xlabel('eps')
	plt.ylabel('min_samples')
	plt.colorbar()
	plt.xticks(np.arange(len(eps_range)), eps_range, rotation=45)
	plt.yticks(np.arange(len(min_samples_range)), min_samples_range)
	plt.title('Distance in meters for DBSCAN')
	plt.show()
	# def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
	# 	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	# 	plt.title(title)
	# 	plt.colorbar()
	# 	tick_marks = np.arange(len(iris.target_names))
	# 	plt.xticks(tick_marks, iris.target_names, rotation=45)
	# 	plt.yticks(tick_marks, iris.target_names)
	# 	plt.tight_layout()
	# 	plt.ylabel('True label')
	# 	plt.xlabel('Predicted label')
	#
	#
	# 	# Compute confusion matrix
	# 	cm = confusion_matrix(y_test, y_pred)
	# 	np.set_printoptions(precision=2)
	# 	print('Confusion matrix, without normalization')
	# 	print(cm)
	# 	plt.figure()
	# 	plot_confusion_matrix(cm)