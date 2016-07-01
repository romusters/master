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


import numpy as np
from sklearn.cluster import DBSCAN

users_coordinates = [[5.91636, 51.9445174], [4.479585, 51.02795], [5.77638889, 52.875], [4.507524, 51.930126], [4.70594, 50.87882], None, [5.99083333, 52.21583333], [4.28306937, 52.01467033], [6.12409, 52.52093], [5.71666667, 52.85], [5.86667, 51.8333], None, None, [6.6017098, 52.2284203], None, [6.5704274, 52.92094747], None, [5.11314869, 52.77242121], None, [5.03085416, 51.62399701], [5.4188469, 50.954919], [3.26305946, 50.83124217], [3.1232688, 50.8029045], None, None, None, [6.089741, 52.512205], None, None, [4.42114949, 51.21638693], [4.94359443, 52.36276197], [3.8904918, 51.2004714], [5.1192979, 52.0921961], None, [5.194456, 50.82204], None, [6.88611111, 53.10472222], [4.88725586, 52.39398971], None, [5.81989, 51.8404], [4.97450313, 51.83154475], None, None]
followers_coordinates = [[], [[4.5, 51.9167], [77.545513, 12.9251151], [5.8705655, 52.8101945]], [[5.70583333, 51.54027778], [5.9769841, 52.2876925], [11.573544, 48.148024], [5.7112887, 52.8447084], [-74.2610792, 40.74620755], [146.98805556, -43.12166667], [4.76805556, 52.67111111], [5.4464726, 52.5194376], [4.657549, 52.484567], [5.06000625, 52.509178]], [[6.48333333, 52.73333333]], [], [[4.29522243, 50.8142068], [2.9605401, 51.2226105], [2.74523284, 50.97942068], [-80.1418, 26.1358], [3.73148172, 51.04986943], [-80.19094, 25.76763], [3.28200976, 50.82940704], [4.893, 52.373], [-4.36667, 50.6667], [4.4867902, 52.1570892], [-84.38795932, 33.76225052], [3.13333, 50.95], [-121.92441, 36.55378], [3.2660799, 50.829731], [6.7, 52.9]], [[5.40188366, 52.1706371], [6.07054565, 52.43992125], [5.99653287, 52.24437445], [5.9571941, 52.2014449], [12.44138889, 43.93833333], [5.77638889, 52.875], [5.941599, 52.2126306]], [[5.0318393, 52.0151777]], [[4.22277, 51.19548]], [[5.70583333, 51.54027778], [146.98805556, -43.12166667], [11.573544, 48.148024], [6.3099676, 51.8526374], [5.77638889, 52.875]], [[4.35, 50.85], [3.72250175, 51.05416944], [3.35564141, 50.88678188], [4.35536479, 50.84879995], [2.07833333, 41.29694444]], [], [[4.513894, 52.154808]], [[6.1369765, 53.0092403]], [[4.49867517, 51.2538417], [5.0804397, 51.266683], [4.48790211, 51.02264878], [4.46279777, 51.26640185], [4.48575877, 51.28686859], [4.40236928, 51.31059798]], [[4.9125443, 51.08689931], [10.9233777, 43.5464955]], [], [[4.78989735, 51.82767287]], [[14.43146887, 50.06299018], [-3.68832707, 40.45311522], [4.3167, 52.0833], [5.05922545, 51.67932606]], [], [[5.18333333, 50.8], [5.960379, 50.891208], [5.47378591, 51.019288], [-87.64908704, 41.92870566]], [[6.9649829, 52.2209412]], [[3.56367758, 51.18211949]], [], [], [], [[6.1342219, 52.5228309], [6.46667, 52.3667], [5.36253664, 52.15651534], [6.089741, 52.512205]], [], [], [[3.27547698, 50.84661193], [2.66784984, 51.11249924], [2.97022772, 51.15608727], [2.59285305, 51.09316517], [2.68853689, 51.11551609], [2.87747538, 51.1284535], [2.74160361, 51.12965425], [2.91329962, 51.04840765], [2.67193438, 51.13178328], [2.59286492, 51.09327511], [3.21386854, 51.19743782], [2.58126469, 51.1008754], [4.62903586, 51.14797411], [2.58176567, 51.09600775], [2.63513384, 51.12032408], [4.31772803, 51.01242473], [3.27036381, 50.82656872], [2.59286492, 51.09327511], [3.22352171, 51.20919057]], [[4.92195188, 52.3625179]], [[4.08143699, 50.93423724]], [[5.147099, 51.58467]], [], [], [], [[7.0817089, 53.0823193]], [], [], [[5.44936111, 51.56765918], [5.70541858, 51.81239046]], [[4.9524376, 51.8166812]], [], []]

#reverse the coordinates
r_users_coordinates = []
r_followers_coordinates = []
for e in users_coordinates:
	if e is None:
		r_users_coordinates.append(None)
	else:
		[a,b] = e
		r_users_coordinates.append([b,a])

for e in followers_coordinates:
	if len(e) == 0:
		r_followers_coordinates.append([])
	else:
		tmp_list = []
		for f in e:
			if len(e) == 0:
				tmp_list.append([])
			else:
				[a,b] = f
				tmp_list.append([b,a])
		r_followers_coordinates.append(tmp_list)

users_coordinates = r_users_coordinates
followers_coordinates = r_followers_coordinates

#filter the empty user or followercoordinates
f_users_coordinates = []
f_followers_coordinates = []
no_user_coordinates = 0
no_follower_coordinates = 0
for i, e in enumerate(users_coordinates):
	if e is not None:
		if len(followers_coordinates[i]) > 0:
			f_users_coordinates.append(e)
			f_followers_coordinates.append(followers_coordinates[i])
		else:
			no_follower_coordinates +=1
	else:
		no_user_coordinates += 1

print "No follower coordinates: ", no_follower_coordinates
print "No user coordinates: ", no_user_coordinates

#eps_range = [0.1, 0.5, 1, 2, 3]
eps_range = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45]
min_samples_range = range(1, 6, 1)

eps_range = []
for e in range(1,101, 1):
	eps_range.append(e*0.1)

min_samples_range = range(1, 6, 1)

result = []
average_results = []
opt_param = None
opt_dist = 99999999999
opt_distances = None
for eps_ in eps_range:
	for min_samples_ in min_samples_range:
		# calculate distance of users to followers for which dbscan predicted the cluster center
		f_u_distances = []
		for i, X in enumerate(f_followers_coordinates):
			# Compute DBSCAN
			X = f_followers_coordinates[i]
			db = DBSCAN(eps=eps_, min_samples=min_samples_).fit(X)
			core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
			core_samples_mask[db.core_sample_indices_] = True
			labels = db.labels_

			# Number of clusters in labels, ignoring noise if present.
			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

			# Black removed and is used for noise instead.
			unique_labels = set(labels)
			max_cluster = 0
			cluster = 0
			class_member_mask = None
			for l in unique_labels:
				count = list(labels).count(l)
				if count > max_cluster:
					max_cluster = count
					cluster = l
					class_member_mask = (labels == l)
			locations = np.asarray(X)[class_member_mask]

			from geopy.distance import vincenty
			distances = []
			for e in locations:
				#distance follower to user
				distance = vincenty(list(e), f_users_coordinates[i]).kilometers
				distances.append(distance)
			# add mean of distances from user to followers
			f_u_distances.append(np.mean(distances))


		# average distance of users to its followers
		u_f_mean = np.mean(f_u_distances)
		if u_f_mean < opt_dist:
			opt_param = (eps_, min_samples_)
			opt_dist = u_f_mean
			opt_distances =  f_u_distances

		# if u_f_mean == 87.561959375300532: #the optimal average distances, this way I select the optimal parameter
		# 	print eps_, min_samples_ #0.5, 1
		# 	optimal_distances_db = f_u_distances
		#distances of users to followers under a certain DBSCAN parameter
		result.append(u_f_mean)

print opt_param

tmp = result[:]
tmp.sort()
print "Sorted distances for DBSCAN: ", tmp

#print optimal_distances_db

#distances for the center of mass approach
distances_com = []
for i, e in enumerate(f_users_coordinates):
	distances_com.append(vincenty(e, np.mean(np.asarray(f_followers_coordinates[i]))).kilometers)


def plot_location_accuracy():
	#convert kilometers to miles
	miles_distances = [e*0.621371192 for e in opt_distances]
	print miles_distances
	miles_distances = [61.447458594003066, 33.95314219590632, 100.43622300911498, 5.422384731740756, 31.942664003322147, 122.5571382563151, 90.55617923942582, 94.46050231535273, 57.42311325179787, 145.1925438334583, 66.74311255908748, 9.956160470459357, 186.2864579173483, 32.50076537525792, 6.951209374567613, 70.27824663920053, 0.9162596799084267, 20.195944304962087, 35.10873160259828, 8.288204966580338, 14.97381047109556, 1.3962867679902444]
	#range for the geolocation distance article
	range_ =  [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
	accuracy_dict = {}
	for e in range_:
		accuracy_dict[str(e)] = 0
		for d in miles_distances:
			if d < e:
				accuracy_dict[str(e)] += 1
	print accuracy_dict
	accuracy = []
	for e in range_:
		accuracy.append((accuracy_dict[str(e)]/len(miles_distances))*100)

	tmp_acc = []
	for e in range(0, 101, 1):
			if e % 10 == 0:
				tmp_acc.append(str(e*1))
			else:
				tmp_acc.append("")

	article_acc = [34, 42, 45, 50, 53, 55, 60, 66, 73, 77, 100]
	import matplotlib.pyplot as plt
	import numpy as np
	plt.figure(figsize=(12,8))
	plt.plot(accuracy, label="Dbscan")
	plt.plot(article_acc, label="RBPDM (K=5)")
	plt.legend()
	plt.title("Geo-location accuracy for different ranges for location disabled users")
	plt.xlabel("Error distance (miles)")
	plt.ylabel("Accuracy (%)")
	plt.xticks(np.arange(len(range_)), tuple(range_))
	plt.yticks(np.arange(101), tuple(tmp_acc))
	plt.savefig("/home/cluster/Dropbox/Master/results/accuracy_distances.png")
	plt.show()

from matplotlib.colors import Normalize
import  matplotlib.pyplot as plt
class MidpointNormalize(Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))

def plot_db_params():
	scores = np.array(result).reshape(len(eps_range), len(min_samples_range))
	plt.figure(figsize=(12, 8))
	plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
			   norm=MidpointNormalize(vmin=0.2, midpoint=0.92), aspect="auto")
	plt.text(0, 50, "Optimum\n(" + str(opt_param[1]) + ", " + str(opt_param[0]) + ")")
	plt.xlabel('min_samples')
	plt.ylabel('eps')
	plt.colorbar()
	tmp_range = []
	for e in range(1, 101, 1):
		if e % 10 == 0:
			tmp_range.append(str(e*0.01))
		else:
			tmp_range.append("")
	#plt.yticks(np.arange(len(eps_range)), eps_range, rotation=45)
	plt.locator_params(axis='y', nbins=10)
	plt.yticks(np.arange(100), tuple(tmp_range), rotation=45)
	plt.xticks(np.arange(len(min_samples_range)), min_samples_range)
	plt.title('Average distance from true to predicted location in km for DBSCAN')


	plt.savefig("/home/cluster/Dropbox/Master/results/geo_dbscan.png")
	plt.show()

def plot_geo_distances():
	plt.figure(figsize=(12, 8))
	plt.plot(distances_com, label="Center of mass approach", color="blue")
	plt.plot(optimal_distances_db, label="Dbscan approach", color="magenta")

	plt.title("Comparison between \"center of mass\" and \"dbscan\" approach")
	plt.xlabel("User")
	plt.ylabel("Distance in kilometers")
	plt.axhline(np.mean(distances_com), color="red", label="Average distance for center of mass", )
	plt.axhline(np.mean(optimal_distances_db), color="green", label="Average distance for dbscan", )
	plt.text(20, 6100, str("Distance=" + "{0:.2f}".format(np.mean(distances_com))))
	plt.text(20, 200, str("Distance=" + "{0:.2f}".format(np.mean(optimal_distances_db))))
	plt.legend(loc=5)
	plt.savefig("/home/cluster/Dropbox/Master/results/geo_distances.png")
	plt.show()

def plot_geo_distances_db():
	import matplotlib.ticker as mtick
	mean = np.mean(optimal_distances_db)
	plt.figure(figsize=(12,8))
	plt.plot(optimal_distances_db, label="Distances per user")
	plt.text(16, 120, str("Distance=" + "{0:.2f}".format(float(mean))))
	plt.title("Geo-location distances for location disabled users with optimal DBSCAN model")
	plt.xlabel("User")
	plt.ylabel("Distance in kilometers")

	plt.axhline(mean, color="red", label="Average distance", )

	plt.legend()
	plt.savefig("/home/cluster/Dropbox/Master/results/geo_distances_db.png")
	plt.show()


def plot_stemming():
	vocabsize = [44556848, 34179930, 34179832, 16352900, 341838, 341411, 341411]
	labels = ["url tag", "mention tag", "stopwords tag", "stemmed", "threshold 10", "reduce haha", "reduce hashtag"]
	plt.figure(figsize=(24, 16))
	plt.plot(vocabsize)
	plt.title("Size of the vocabulary after several dimension reduction techniques")
	#plt.ylabel("vocabsize")
	plt.xticks([0,1,2,3,4,5,6], labels, rotation=45)
	#plt.yticks(len(vocabsize), tuple(vocabsize))
	plt.savefig("/home/cluster/Dropbox/Master/results/vocabsize.png")
	plt.show()
#plot_db_params()

plot_location_accuracy()
#plot_geo_distances()
#plot_geo_distances_db()