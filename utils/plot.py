#Plot the coordinates of tweets on a map.
#Future: map coordinates on the map of the Netherlands

import numpy as np
from scipy.optimize import curve_fit

# def plot_coordinates(coordinates):
# 	plt.plot(*zip(*coordinates))

def plot_model_size(folder):
	x= [0.5, 1, 2, 4, 8]

	data = [0.46, 0.77, 1.3, 2.1, 3.4]

	popt, pcov = curve_fit(func, x, data)

	import matplotlib.pyplot as plt

	limit = 16
	x2 = np.arange(0, limit, 1)
	y2 = x2 * popt[0] + popt[1]


	fig = plt.figure()

	ax = fig.add_subplot(111)

	ax.errorbar(x[0], data[0], yerr=y2[0]-data[0], ecolor='b', fmt='bo')
	ax.errorbar(x[1], data[1], yerr=y2[1]-data[1], ecolor='b', fmt='bo')
	ax.errorbar(x[2], data[2], yerr=y2[2]-data[2], ecolor='b', fmt='bo')
	ax.errorbar(x[3], data[3], yerr=y2[3]-data[3], ecolor='b', fmt='bo')
	ax.errorbar(x[4], data[4], yerr=y2[4]-data[4], ecolor='b', fmt='bo')
	plt.title("Amount of storage used for the model per Gb of data")
	plt.plot(x, data, 'b', x2, y2, 'g-')
	#plt.plot(x, time)
	plt.ylabel('Storage (Gb)')
	plt.xlabel('Data (Gb)')
	plt.savefig(folder + "model.png")
	plt.show()

def plot_word_freq(fname):
	occs = []
	import matplotlib.pyplot as plt
	for line in open(fname, 'r'):
		data = eval(line)
		occs.append(data[1])
		print data
	x = range(1, len(occs))
	plt.title("Occurrences")
	plt.plot(x, occs)
	plt.ylabel('Occurrences')
	plt.xlabel('Feature index')
	#plt.savefig(folder + "time.png")
	plt.show()


def plot_time(folder):
	import matplotlib.pyplot as plt
	x = [0.5, 1, 2, 4, 8]
	data = [6*60+39, 19*60+38, 23*60+3, 49*60+18, 93*60+33]

	popt, pcov = curve_fit(func, x, data)

	limit = 16
	x2 = np.arange(0, limit, 1)
	y2 = x2 * popt[0] + popt[1]

	fig = plt.figure()

	ax = fig.add_subplot(111)

	ax.errorbar(x[0], data[0], yerr=y2[0]-data[0], ecolor='b', fmt='bo')
	ax.errorbar(x[1], data[1], yerr=y2[1]-data[1], ecolor='b', fmt='bo')
	ax.errorbar(x[2], data[2], yerr=y2[2]-data[2], ecolor='b', fmt='bo')
	ax.errorbar(x[3], data[3], yerr=y2[3]-data[3], ecolor='b', fmt='bo')
	ax.errorbar(x[4], data[4], yerr=y2[4]-data[4], ecolor='b', fmt='bo')

	plt.title("Amount of time used per Gb of data")
	plt.plot(x, data, 'b', x2, y2, 'g-')
	plt.ylabel('Time (s)')
	plt.xlabel('Data (Gb)')
	plt.savefig(folder + "time.png")
	plt.show()



def plot_ram(folder):
	import matplotlib.pyplot as plt
	x = [0.5, 1, 2, 4, 8]
	data = [2.02, 3.8, 5.99, 10.91, 18.84]

	popt, pcov = curve_fit(func, x, data)

	limit = 16
	x2 = np.arange(0, limit, 1)
	y2 = x2 * popt[0] + popt[1]

	fig = plt.figure()

	ax = fig.add_subplot(111)

	ax.errorbar(x[0], data[0], yerr=y2[0]-data[0], ecolor='b', fmt='bo')
	ax.errorbar(x[1], data[1], yerr=y2[1]-data[1], ecolor='b', fmt='bo')
	ax.errorbar(x[2], data[2], yerr=y2[2]-data[2], ecolor='b', fmt='bo')
	ax.errorbar(x[3], data[3], yerr=y2[3]-data[3], ecolor='b', fmt='bo')
	ax.errorbar(x[4], data[4], yerr=y2[4]-data[4], ecolor='b', fmt='bo')

	plt.title("Amount of RAM used per Gb of data")
	plt.plot(x, data, 'b', x2, y2, 'g-')
	plt.ylabel('RAM (Gb)')
	plt.xlabel('Data (Gb)')
	plt.savefig(folder + "ram.png")
	plt.show()


#for five years of tweets, plot the average frequency per day
def plot_dates(year):
	import sys
	from datetime import datetime
	fname = "/home/robert/dates"
	first_day = datetime(year, 1, 1, 0, 0)

	day_freqs = {}
	days = range(1,365,1)
	for i in days:
		day_freqs[str(i)] = 0

	with open(fname) as f:
		for line in f:
			data = eval(line)
			date = datetime.strptime(data[0], '(%Y, %m, %d)')
			index = date - first_day
			day_freqs[str(index.days)] = data[1]

	print len(day_freqs)
	return day_freqs

def plot_dates_fixed(year, freqs, folder):
	# year_2011 = {'344': 189177, '345': 179646, '346': 165493, '347': 150477, '340': 132423, '341': 131069, '342': 129730, '343': 103098, '348': 158868, '349': 135458, '298': 171239, '299': 203471, '296': 373334, '297': 153596, '294': 256036, '295': 265041, '292': 191452, '293': 201450, '290': 190477, '291': 266299, '199': 317043, '198': 301468, '195': 322531, '194': 257480, '197': 334707, '196': 286560, '191': 324661, '190': 333213, '193': 240535, '192': 312842, '270': 195816, '271': 182280, '272': 44815, '273': 75286, '274': 174050, '275': 245468, '276': 156033, '277': 170140, '278': 112282, '279': 105918, '108': 245712, '109': 126403, '102': 163355, '103': 161700, '100': 259909, '101': 262983, '106': 224859, '107': 243541, '104': 154745, '105': 192375, '39': 357102, '38': 386323, '33': 279479, '32': 398277, '31': 411240, '30': 408832, '37': 348111, '36': 297765, '35': 358485, '34': 462138, '339': 140496, '338': 196997, '335': 119470, '334': 125981, '337': 202876, '336': 127776, '331': 277536, '330': 134906, '333': 114688, '332': 140201, '6': 612710, '99': 264771, '98': 191970, '91': 88641, '90': 96093, '93': 215769, '92': 118512, '95': 163231, '94': 103472, '97': 187862, '96': 136166, '238': 258047, '239': 341051, '234': 327990, '235': 247410, '236': 352500, '237': 252750, '230': 334252, '231': 344263, '232': 260210, '233': 317111, '1': 1220415, '146': 153088, '147': 118381, '144': 150396, '145': 127272, '142': 287896, '143': 140870, '140': 120042, '141': 240867, '148': 236018, '149': 286128, '133': 196185, '132': 151188, '131': 99039, '130': 94014, '137': 158763, '136': 177425, '135': 295429, '134': 153939, '139': 128868, '138': 121683, '24': 667980, '25': 424174, '26': 514571, '27': 323928, '20': 612462, '21': 212036, '22': 426884, '23': 382083, '28': 393969, '29': 310579, '88': 136555, '89': 111657, '82': 113987, '83': 163164, '80': 164815, '81': 136055, '86': 239313, '87': 173570, '84': 116468, '85': 173687, '7': 679575, '245': 184549, '244': 172563, '247': 212438, '246': 274299, '241': 263876, '240': 307081, '243': 220829, '242': 269434, '249': 184935, '248': 170415, '179': 144400, '178': 113813, '177': 298360, '176': 242131, '175': 144007, '174': 196496, '173': 562242, '172': 57636, '171': 756, '170': 0, '253': 227491, '182': 163481, '183': 295681, '180': 152378, '181': 216610, '186': 236088, '187': 216849, '184': 319373, '185': 313195, '188': 273345, '189': 241581, '11': 616597, '10': 708799, '13': 798037, '12': 683468, '15': 1138438, '14': 659106, '17': 482542, '16': 490349, '19': 594877, '18': 522863, '62': 175673, '322': 85288, '323': 221883, '320': 116992, '321': 105626, '326': 127186, '327': 118450, '324': 256673, '325': 101524, '328': 112716, '329': 116127, '201': 279660, '200': 222051, '203': 294373, '202': 385704, '205': 388655, '204': 392244, '207': 361997, '206': 280915, '209': 368626, '208': 367423, '77': 156549, '76': 160472, '75': 160968, '74': 103978, '73': 113990, '72': 161574, '71': 155361, '70': 128739, '79': 277686, '78': 191225, '2': 977146, '8': 731201, '68': 144803, '120': 373581, '121': 273309, '122': 255677, '123': 192601, '124': 202530, '125': 232694, '126': 238517, '127': 187, '128': 243468, '129': 153664, '319': 137379, '318': 210677, '313': 82685, '312': 107858, '311': 105554, '310': 246892, '317': 247526, '316': 264697, '315': 127690, '314': 118155, '3': 844493, '364': 262379, '365': 144910, '362': 226495, '363': 178063, '360': 188061, '361': 241938, '60': 151202, '61': 152737, '258': 198017, '259': 163162, '64': 196074, '65': 234605, '66': 179344, '67': 161030, '252': 137482, '69': 134467, '250': 141282, '251': 165723, '256': 183142, '257': 192542, '254': 231410, '255': 228991, '168': 138067, '169': 185711, '164': 131260, '165': 120922, '166': 90662, '167': 166717, '160': 167679, '161': 148822, '162': 255236, '163': 217319, '9': 929172, '357': 122346, '356': 143782, '355': 131767, '354': 126886, '353': 134996, '352': 170039, '351': 243912, '350': 127696, '359': 156839, '358': 243312, '216': 313321, '217': 384058, '214': 315076, '215': 341259, '212': 288988, '213': 352750, '210': 276226, '211': 290163, '218': 424382, '219': 394895, '289': 310263, '288': 193917, '4': 770059, '281': 197995, '280': 123319, '283': 383607, '282': 238086, '285': 628966, '284': 371868, '287': 176836, '286': 137925, '263': 153074, '262': 223638, '261': 244750, '260': 171308, '267': 224626, '266': 172908, '265': 116707, '264': 160661, '269': 148021, '268': 289617, '59': 147938, '58': 193047, '55': 167558, '54': 59645, '57': 182202, '56': 92331, '51': 106167, '50': 88879, '53': 60024, '52': 82572, '63': 185034, '115': 229110, '114': 214933, '117': 115062, '116': 230005, '111': 175241, '110': 130056, '113': 290079, '112': 255437, '119': 66368, '118': 120410, '308': 135032, '309': 288903, '300': 206732, '301': 159444, '302': 251749, '303': 240371, '304': 127568, '305': 123535, '306': 70283, '307': 118171, '229': 343314, '228': 375460, '227': 338638, '226': 309280, '225': 261771, '224': 345010, '223': 337952, '222': 330980, '221': 309910, '220': 339551, '151': 42981, '150': 145954, '153': 136293, '152': 129958, '155': 264425, '154': 223919, '157': 172383, '156': 273477, '159': 134308, '158': 158807, '48': 68393, '49': 48714, '46': 147937, '47': 75651, '44': 229915, '45': 247204, '42': 191542, '43': 154099, '40': 234468, '41': 251762, '5': 543219}
	# year_2012 = {}
	# year_2015 = {'344': 0, '345': 0, '346': 0, '347': 0, '340': 0, '341': 0, '342': 0, '343': 0, '348': 0, '349': 0, '298': 626869, '299': 689946, '296': 747953, '297': 748725, '294': 658601, '295': 778028, '292': 647233, '293': 726043, '290': 766938, '291': 630953, '199': 653563, '198': 666918, '195': 694670, '194': 559593, '197': 696027, '196': 692305, '191': 706816, '190': 736002, '193': 547747, '192': 667598, '270': 621869, '271': 655585, '272': 775874, '273': 743114, '274': 770032, '275': 774710, '276': 732893, '277': 611075, '278': 669459, '279': 749511, '108': 814354, '109': 729079, '102': 740602, '103': 678559, '100': 824866, '101': 803912, '106': 819429, '107': 830503, '104': 774783, '105': 819918, '39': 748236, '38': 826598, '33': 858439, '32': 758621, '31': 868994, '30': 1120636, '37': 887651, '36': 899729, '35': 885371, '34': 809120, '339': 0, '338': 0, '335': 0, '334': 599475, '337': 0, '336': 0, '331': 782060, '330': 406825, '333': 658607, '332': 781273, '6': 837745, '99': 815780, '98': 804394, '91': 935586, '90': 837596, '93': 895209, '92': 885785, '95': 696184, '94': 812839, '97': 677847, '96': 658790, '238': 767074, '239': 387770, '234': 698881, '235': 610453, '236': 653443, '237': 778605, '230': 712401, '231': 738949, '232': 751619, '233': 752414, '1': 0, '146': 692503, '147': 768631, '144': 807673, '145': 601385, '142': 844923, '143': 749618, '140': 819267, '141': 836556, '148': 812023, '149': 790443, '133': 797371, '132': 774748, '131': 663657, '130': 688197, '137': 660809, '136': 714021, '135': 679683, '134': 773528, '139': 806894, '138': 736486, '24': 810387, '25': 781764, '26': 798020, '27': 840880, '20': 843526, '21': 861830, '22': 842512, '23': 844378, '28': 842451, '29': 903117, '88': 795966, '89': 738093, '82': 904526, '83': 869407, '80': 893674, '81': 815579, '86': 875999, '87': 700581, '84': 945719, '85': 901911, '7': 729980, '245': 788237, '244': 858011, '247': 836631, '246': 781518, '241': 780473, '240': 653279, '243': 754956, '242': 664760, '249': 649582, '248': 757549, '179': 612063, '178': 755734, '177': 737860, '176': 811010, '175': 750302, '174': 779642, '173': 690019, '172': 683267, '171': 768750, '170': 800622, '253': 766696, '182': 783516, '183': 832302, '180': 609167, '181': 775462, '186': 626292, '187': 657906, '184': 840127, '185': 735899, '188': 722775, '189': 715114, '11': 746765, '10': 877488, '13': 828426, '12': 783644, '15': 913976, '14': 831265, '17': 805609, '16': 830958, '19': 786504, '18': 753228, '62': 847090, '322': 869568, '323': 805857, '320': 731716, '321': 795832, '326': 702445, '327': 746809, '324': 772543, '325': 811254, '328': 778309, '329': 220758, '201': 529065, '200': 508959, '203': 629073, '202': 644797, '205': 629573, '204': 624141, '207': 616944, '206': 619630, '209': 618512, '208': 560696, '77': 931338, '76': 850412, '75': 813897, '74': 776613, '73': 854272, '72': 791314, '71': 997245, '70': 937783, '79': 917247, '78': 1008707, '2': 713495, '8': 940787, '68': 784355, '120': 788833, '121': 779544, '122': 782095, '123': 632333, '124': 733411, '125': 779048, '126': 756607, '127': 804935, '128': 781151, '129': 733320, '319': 874198, '318': 915190, '313': 710430, '312': 663081, '311': 977399, '310': 799773, '317': 792503, '316': 804658, '315': 751921, '314': 751576, '3': 754002, '364': 0, '365': 0, '362': 0, '363': 0, '360': 0, '361': 0, '60': 740118, '61': 829692, '258': 735140, '259': 811167, '64': 867257, '65': 889302, '66': 815534, '67': 740170, '252': 768666, '69': 918028, '250': 764583, '251': 760781, '256': 669554, '257': 653282, '254': 738098, '255': 718658, '168': 807565, '169': 804062, '164': 768831, '165': 654317, '166': 664306, '167': 775010, '160': 731091, '161': 767978, '162': 796958, '163': 820345, '9': 880447, '357': 0, '356': 0, '355': 0, '354': 0, '353': 0, '352': 0, '351': 0, '350': 0, '359': 0, '358': 0, '216': 603792, '217': 655232, '214': 495097, '215': 502576, '212': 611871, '213': 591350, '210': 622925, '211': 662983, '218': 640186, '219': 643574, '289': 773527, '288': 770952, '4': 711976, '281': 799883, '280': 798138, '283': 746398, '282': 755450, '285': 635140, '284': 695177, '287': 854855, '286': 735947, '263': 647074, '262': 763529, '261': 816655, '260': 797050, '267': 816786, '266': 775571, '265': 724732, '264': 654211, '269': 739097, '268': 778099, '59': 856613, '58': 942319, '55': 919920, '54': 831632, '57': 922924, '56': 941889, '51': 898544, '50': 847754, '53': 755067, '52': 866371, '63': 862618, '115': 784668, '114': 822321, '117': 709700, '116': 682634, '111': 805058, '110': 754928, '113': 870182, '112': 817587, '119': 727639, '118': 658868, '308': 799541, '309': 791880, '300': 732747, '301': 749559, '302': 817213, '303': 737513, '304': 746939, '305': 607338, '306': 652403, '307': 719090, '229': 607404, '228': 589957, '227': 692348, '226': 702043, '225': 681453, '224': 625308, '223': 655954, '222': 542676, '221': 541030, '220': 631638, '151': 671157, '150': 766572, '153': 778489, '152': 720979, '155': 793495, '154': 810946, '157': 785674, '156': 791982, '159': 637712, '158': 648717, '48': 793374, '49': 836722, '46': 737235, '47': 723169, '44': 857575, '45': 822621, '42': 850369, '43': 883526, '40': 792234, '41': 842728, '5': 764289}

	keys = [int(e) for e in freqs.keys()]
	keys.sort()

	values = []
	for key in keys:
		values.append(freqs[str(key)])

	import matplotlib.pyplot as plt

	x = range(1, 366, 1)
	y = [v/1000 for v in values]
	plt.title("Amount of Tweets send per day in the year " + str(year))
	plt.ylabel('Number of Tweets *10^3')
	plt.xlabel('Day')
	plt.xlim([1,365])

	plt.plot(x, y)
	plt.savefig(folder + "tweets_hadoop.png")
	plt.show()



def func(x, a, b):
	return a * x + b

def plot_coordinates():
	import utils
	import validateGeo
	coordinates = [[5.40188366, 52.1706371], [6.07054565, 52.43992125], [5.99653287, 52.24437445], [5.9571941, 52.2014449], [12.44138889, 43.93833333], [5.77638889, 52.875], [5.941599, 52.2126306]]
	username = "198488640"
	# user = utils.Utils(username)
	# user_coordinates = user.getCoordinate(username)
	user_coordinates = [5.99083333, 52.21583333]
	print user_coordinates
	print validateGeo.center_geolocation(coordinates)


def plot_occurrences():
	fname = '/home/cluster/counts_taggedUrl_Mention_Stopwords_Punctuation_ignoreNonAscii_Stemmed'
	occs = []
	# with open(fname, 'r') as f:
	# 	for line in f:
	# 		value = eval(line)[1]
	# 		occs.append(value)
	#
	# with open('/home/cluster/occurrences.txt', 'w+') as f:
	# 	for e in occs:
	# 		f.write("%s\n" % e)
	import math
	with open('/home/cluster/occurrences.txt', 'r') as f:
		for line in f:
			occs.append(math.log(int(line)))

	x = range(0, len(occs), 1)
	import matplotlib.pyplot as plt
	plt.plot(x, occs)
	plt.show()


def plot_distances(path, resultsfolder):
	import csv
	distances= []
	tweets = []
	with open(path, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			distances.append(eval(row[4]))
			tweets.append(row[0])

	#calculate the derivative
	if 1:
		dx = 4
		dy = []
		for i, e in enumerate(distances):
			distances[i] = e+1
		dy = np.convolve([-1,0,0,1], distances)/dx

	#create a mask
	if 1:
		cluster = []
		for i, e in enumerate(dy):
			if dy[i] == 0:
				dy[i] = 1
				cluster.append(tweets[i])
			else:
				if len(cluster) < 20:
					continue
				#print cluster
				dy[i] = 0
		distances = dy


	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(12,8))
	ax = fig.add_axes((.1,.4,.8,.5))
	ax.plot(distances)
	plt.title("Distance from an average tweet in data to the other tweets.")
	plt.xlabel("Tweet ID")
	plt.ylabel("Soft max Cosine similarity")
	txt =  """The dataset consists of 1% of the first month of 2015. The Word2vec model is trained on all words from the first month of 2015.
The output layer of the model is 718. The distance of a randomly chosen tweet is calculated to all the other tweets.
The distances are sorted and shown as the blue line in the figure above.
The distance between two tweets is calculated as follows: Every word from tweet A is compared to a word from tweet B.
The word with the highest similarity, or distance, is added to the list. The list is averaged with the amount of words in tweet B."""
	fig.text(.1,.1, txt)
	plt.savefig(results_folder + "average_tweet_distance_w2v_derivative_mask.png")
	#plt.savefig(results_folder + "average_tweet_distance_w2v.png")
	plt.show()

	return result

def plot_kmeans_for_w2v(path):
	distances = []
	xrange = []
	fname = "kmeans_for_w2v"
	with open(path + fname) as f:
		for line in f:
			line = line.rstrip('\n')
			line = line .split("&")
			distances.append(eval(line[1]))
			xrange.append(eval(line[0]))
	import matplotlib.pyplot as plt
	plt.plot(xrange, distances)
	plt.title("Kmeans on w2v vectors")
	plt.xlabel("Number of clusters")
	plt.ylabel("Within Set Sum of Squared Error")
	plt.savefig(path + fname + ".png")
	plt.show()
	print distances


def plot_kmeans_for_lda(path):
	distances = []
	xrange = []
	fname = "kmeans_for_lda"
	with open(path + fname) as f:
		for line in f:
			line = line.rstrip('\n')
			line = line .split("&")
			distances.append(eval(line[1]))
			xrange.append(eval(line[0]))
	import matplotlib.pyplot as plt
	plt.plot(xrange, distances)
	plt.title("Kmeans on LDA vectors")
	plt.xlabel("Number of clusters")
	plt.ylabel("Within Set Sum of Squared Error")
	plt.savefig(path + fname + ".png")
	plt.show()


if __name__ == "__main__":
	#main()
	results_folder = '/home/robert/Dropbox/Master/results/'
	results_folder = '/home/cluster/Dropbox/Master/results/'
	plot_kmeans_for_w2v(results_folder)

	#plot_cpu_scaling()
	#plot_ram(results_folder)
	#plot_time(results_folder)
	#plot_model_size(results_folder)
	#plot_coordinates()

	year = 2015
	#freqs = plot_dates(year)
	#plot_dates_fixed(year, freqs, results_folder)
	#plot_word_freq('/home/robert/counts_sorted')

	#plot_occurrences()
	path = 	"/home/cluster/nd.csv"
	#plot_distances(path, results_folder)
	path = "/home/cluster/Dropbox/Master/results/"
	plot_kmeans_for_lda(path)