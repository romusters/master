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
	import dateutil.parser
	import sys
	from datetime import date

	#fname = "/media/robert/dataThesis/tweets/dates.txt"
	fname = "/home/robert/Downloads/dates.txt"
	last_day = date(year, 12, 31)
	day_freqs = {}
	days = range(1,366,1)
	for i in days:
		day_freqs[str(i)] = 0

	old_day_number = -1
	with open(fname) as f:
		for line in f:
			date = dateutil.parser.parse(line)
			if date.year == year:
				day_number = (365 + (date.date() - last_day).days) + 1 #+1 is correct?
				if day_number == old_day_number:
					day_freqs[str(day_number)] += 1
					pass
				else:
					print day_number
					old_day_number = day_number
					day_freqs[str(day_number)] += 1
			if date.year == year +1:
				print day_freqs
				sys.exit(0)

	print day_freqs

def plot_dates_fixed(folder):
	year_2011 = {'344': 189177, '345': 179646, '346': 165493, '347': 150477, '340': 132423, '341': 131069, '342': 129730, '343': 103098, '348': 158868, '349': 135458, '298': 171239, '299': 203471, '296': 373334, '297': 153596, '294': 256036, '295': 265041, '292': 191452, '293': 201450, '290': 190477, '291': 266299, '199': 317043, '198': 301468, '195': 322531, '194': 257480, '197': 334707, '196': 286560, '191': 324661, '190': 333213, '193': 240535, '192': 312842, '270': 195816, '271': 182280, '272': 44815, '273': 75286, '274': 174050, '275': 245468, '276': 156033, '277': 170140, '278': 112282, '279': 105918, '108': 245712, '109': 126403, '102': 163355, '103': 161700, '100': 259909, '101': 262983, '106': 224859, '107': 243541, '104': 154745, '105': 192375, '39': 357102, '38': 386323, '33': 279479, '32': 398277, '31': 411240, '30': 408832, '37': 348111, '36': 297765, '35': 358485, '34': 462138, '339': 140496, '338': 196997, '335': 119470, '334': 125981, '337': 202876, '336': 127776, '331': 277536, '330': 134906, '333': 114688, '332': 140201, '6': 612710, '99': 264771, '98': 191970, '91': 88641, '90': 96093, '93': 215769, '92': 118512, '95': 163231, '94': 103472, '97': 187862, '96': 136166, '238': 258047, '239': 341051, '234': 327990, '235': 247410, '236': 352500, '237': 252750, '230': 334252, '231': 344263, '232': 260210, '233': 317111, '1': 1220415, '146': 153088, '147': 118381, '144': 150396, '145': 127272, '142': 287896, '143': 140870, '140': 120042, '141': 240867, '148': 236018, '149': 286128, '133': 196185, '132': 151188, '131': 99039, '130': 94014, '137': 158763, '136': 177425, '135': 295429, '134': 153939, '139': 128868, '138': 121683, '24': 667980, '25': 424174, '26': 514571, '27': 323928, '20': 612462, '21': 212036, '22': 426884, '23': 382083, '28': 393969, '29': 310579, '88': 136555, '89': 111657, '82': 113987, '83': 163164, '80': 164815, '81': 136055, '86': 239313, '87': 173570, '84': 116468, '85': 173687, '7': 679575, '245': 184549, '244': 172563, '247': 212438, '246': 274299, '241': 263876, '240': 307081, '243': 220829, '242': 269434, '249': 184935, '248': 170415, '179': 144400, '178': 113813, '177': 298360, '176': 242131, '175': 144007, '174': 196496, '173': 562242, '172': 57636, '171': 756, '170': 0, '253': 227491, '182': 163481, '183': 295681, '180': 152378, '181': 216610, '186': 236088, '187': 216849, '184': 319373, '185': 313195, '188': 273345, '189': 241581, '11': 616597, '10': 708799, '13': 798037, '12': 683468, '15': 1138438, '14': 659106, '17': 482542, '16': 490349, '19': 594877, '18': 522863, '62': 175673, '322': 85288, '323': 221883, '320': 116992, '321': 105626, '326': 127186, '327': 118450, '324': 256673, '325': 101524, '328': 112716, '329': 116127, '201': 279660, '200': 222051, '203': 294373, '202': 385704, '205': 388655, '204': 392244, '207': 361997, '206': 280915, '209': 368626, '208': 367423, '77': 156549, '76': 160472, '75': 160968, '74': 103978, '73': 113990, '72': 161574, '71': 155361, '70': 128739, '79': 277686, '78': 191225, '2': 977146, '8': 731201, '68': 144803, '120': 373581, '121': 273309, '122': 255677, '123': 192601, '124': 202530, '125': 232694, '126': 238517, '127': 187, '128': 243468, '129': 153664, '319': 137379, '318': 210677, '313': 82685, '312': 107858, '311': 105554, '310': 246892, '317': 247526, '316': 264697, '315': 127690, '314': 118155, '3': 844493, '364': 262379, '365': 144910, '362': 226495, '363': 178063, '360': 188061, '361': 241938, '60': 151202, '61': 152737, '258': 198017, '259': 163162, '64': 196074, '65': 234605, '66': 179344, '67': 161030, '252': 137482, '69': 134467, '250': 141282, '251': 165723, '256': 183142, '257': 192542, '254': 231410, '255': 228991, '168': 138067, '169': 185711, '164': 131260, '165': 120922, '166': 90662, '167': 166717, '160': 167679, '161': 148822, '162': 255236, '163': 217319, '9': 929172, '357': 122346, '356': 143782, '355': 131767, '354': 126886, '353': 134996, '352': 170039, '351': 243912, '350': 127696, '359': 156839, '358': 243312, '216': 313321, '217': 384058, '214': 315076, '215': 341259, '212': 288988, '213': 352750, '210': 276226, '211': 290163, '218': 424382, '219': 394895, '289': 310263, '288': 193917, '4': 770059, '281': 197995, '280': 123319, '283': 383607, '282': 238086, '285': 628966, '284': 371868, '287': 176836, '286': 137925, '263': 153074, '262': 223638, '261': 244750, '260': 171308, '267': 224626, '266': 172908, '265': 116707, '264': 160661, '269': 148021, '268': 289617, '59': 147938, '58': 193047, '55': 167558, '54': 59645, '57': 182202, '56': 92331, '51': 106167, '50': 88879, '53': 60024, '52': 82572, '63': 185034, '115': 229110, '114': 214933, '117': 115062, '116': 230005, '111': 175241, '110': 130056, '113': 290079, '112': 255437, '119': 66368, '118': 120410, '308': 135032, '309': 288903, '300': 206732, '301': 159444, '302': 251749, '303': 240371, '304': 127568, '305': 123535, '306': 70283, '307': 118171, '229': 343314, '228': 375460, '227': 338638, '226': 309280, '225': 261771, '224': 345010, '223': 337952, '222': 330980, '221': 309910, '220': 339551, '151': 42981, '150': 145954, '153': 136293, '152': 129958, '155': 264425, '154': 223919, '157': 172383, '156': 273477, '159': 134308, '158': 158807, '48': 68393, '49': 48714, '46': 147937, '47': 75651, '44': 229915, '45': 247204, '42': 191542, '43': 154099, '40': 234468, '41': 251762, '5': 543219}
	year_2012 = {}

	keys = [int(e) for e in year_2011.keys()]
	keys.sort()

	values = []
	for key in keys:
		values.append(year_2011[str(key)])

	import matplotlib.pyplot as plt

	x = range(1, 366, 1)
	y = [v/1000 for v in values]
	plt.title("Amount of Tweets send per day in the year 2011")
	plt.ylabel('Number of Tweets *10^3')
	plt.xlabel('Day')
	plt.xlim([1,365])

	plt.plot(x, y)
	plt.savefig(folder + "tweets.png")
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

if __name__ == "__main__":
	#main()
	#results_folder = '/home/robert/Dropbox/Master/results/'
	#plot_cpu_scaling()
	#plot_ram(results_folder)
	#plot_time(results_folder)
	#plot_model_size(results_folder)
	#plot_coordinates()

	year = 2015
	plot_dates(year)
	#plot_dates_fixed(results_folder)