#Plot the coordinates of tweets on a map.
#Future: map coordinates on the map of the Netherlands

import numpy as np
from scipy.optimize import curve_fit

def plot_coordinates(coordinates):
	plt.plot(*zip(*coordinates))

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
def plotDates(list_of_date_times):
	import matplotlib
	dates = matplotlib.dates.date2num(list_of_datetimes)
	plot_date(dates, values)


def func(x, a, b):
	return a * x + b


if __name__ == "__main__":
	#main()
	results_folder = '/home/robert/Dropbox/Master/results/'
	#plot_cpu_scaling()
	#plot_ram(results_folder)
	plot_time(results_folder)
	#plot_model_size(results_folder)