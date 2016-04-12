#Validate the outcome of the paper where geolocation disabled tweets get inferred through the user its friends.
import utils
import requests.packages.urllib3
import logging
import sys
import time

#disable ssl warnings
requests.packages.urllib3.disable_warnings()

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def test():
	util = utils.Utils(user)
	coordinateA = util.getCoordinate(util.user)


def main():
	time.sleep(60*15)
	#Load data from a single day
	#fname = "/home/robert/data/2015123101/20151231:01.out"
	fname = "/home/robert/data/20151225to31/20151231:11.out"
	results = "/home/robert/data/geovalidate/"
	#load data which has geolocation information
	data = utils.loadData(fname, True, True)
	data = data[0:100]
	users = data[2]

	#for 100 samples, validate if the inferred location is indeed the correct location
	#samples are stored in the folder "results" under the filename: "geovalidate.ods".
	for user in users:
		f = open(results + str(user), 'w')

		util = utils.Utils(user)
		#if the user exists
		if util.user is not None:
			friends = util.getFriends(user)
			f.write("Length of friends is:\n")
			f.write(str(len(friends)) + '\n')

			coordinates = []
			for friend in friends:
				if friend.__getstate__()['geo_enabled']:
					logger.debug("Geo enabled")
					try:
						status = friend.__getstate__()['status']
					except KeyError:
						logger.debug("No status information.")
						continue

					if status is not None:
						if friend.__getstate__()['status'].__getstate__()['coordinates'] is not None:
							coordinate = friend.__getstate__()['status'].__getstate__()['coordinates']['coordinates']
							if coordinate is not None:
								coordinates.append(coordinate)
						else:
							logger.debug("No coordinate information.")
					else:
						logger.debug("No status information.")
				else:
					logger.debug("No geo information.")

			if coordinates is not None:
				f.write("Amount of coordinates found:\n")
				f.write(str(len(coordinates)) + '\n')
				f.write("The coordinates are:\n" )
				f.write(str(coordinates) + '\n')
			else:
				logger.info("No coordinates found :(")
		f.close()
	logger.info("Sleep")
	time.sleep(60*15)


def inferEachUser(user):

		util = utils.Utils(user)
		coordinateA = util.getCoordinate(util.user)

		coordinates = []
		#limit the amount of users retreived for Twitter rules
		for friend in util.user.friends():
			coordinates.append(util.getCoordinate(friend))
		coordinateB = center_geolocation(coordinates)

		#calculate the distance between two coordinates
		print utils.distance(coordinateA, coordinateB)


def dateFeature(iday):
	import math
	t = iday-1.0

	phi = t/365.0  * (2*math.pi)

	x = math.cos(phi)
	y = math.sin(phi)

	return x, y


#calculate center of mass to check if the inferred location is indeed the correct location
def centerMass():
	import numpy as np
	import matplotlib.pyplot as plt

	#points.toArray
	n = 20
	m = [1] * n
	x = np.random.randint(-50, 50, n)
	y = np.random.randint(0,200,n)

	cgx = np.sum(x*m)/np.sum(m)
	cgy = np.sum(y*m)/np.sum(m)

	plt.rcParams['figure.figsize'] = (6, 10)  # (width, height)
	plt.scatter(x,y,s=m)
	plt.scatter(cgx, cgy, color='k', marker='+', s=1e4)
	plt.title('2 Dimensional Center of Gravity')
	plt.show()

	return cgx, cgy


def center_geolocation(geolocations):
	"""
	Provide a relatively accurate center lat, lon returned as a list pair, given
	a list of list pairs.
	ex: in: geolocations = ((lat1,lon1), (lat2,lon2),)
		out: (center_lat, center_lon)
	"""
	x = 0
	y = 0
	z = 0

	for lat, lon in geolocations:
		lat = float(lat)
		lon = float(lon)
		x += cos(lat) * cos(lon)
		y += cos(lat) * sin(lon)
		z += sin(lat)

	x = float(x / len(geolocations))
	y = float(y / len(geolocations))
	z = float(z / len(geolocations))

	return (atan2(y, x), atan2(z, sqrt(x * x + y * y)))



def alarm():
	import time
	import sys
	import pyaudio
	import math
	for d in range(1,16):
		print d
		time.sleep(60)

	PyAudio = pyaudio.PyAudio

	#See http://en.wikipedia.org/wiki/Bit_rate#Audio
	BITRATE = 16000 #number of frames per second/frameset.

	#See http://www.phy.mtu.edu/~suits/notefreqs.html
	FREQUENCY = 261.63 #Hz, waves per second, 261.63=C4-note.
	LENGTH = 1.2232 #seconds to play sound

	NUMBEROFFRAMES = int(BITRATE * LENGTH)
	RESTFRAMES = NUMBEROFFRAMES % BITRATE
	WAVEDATA = ''

	for x in xrange(NUMBEROFFRAMES):
	 WAVEDATA = WAVEDATA+chr(int(math.sin(x/((BITRATE/FREQUENCY)/math.pi))*127+128))

	#fill remainder of frameset with silence
	for x in xrange(RESTFRAMES):
	 WAVEDATA = WAVEDATA+chr(128)

	p = PyAudio()
	stream = p.open(format = p.get_format_from_width(1),
					channels = 1,
					rate = BITRATE,
					output = True)
	stream.write(WAVEDATA)
	stream.stop_stream()
	stream.close()
	p.terminate()

if __name__ == "__main__":
	main()
