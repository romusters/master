#Validate the outcome of the paper where geolocation disabled tweets get inferred through the user its friends.
import utils
import requests.packages.urllib3
import logging
import sys
import time
import tweepy
from requests.exceptions import Timeout, ConnectionError
from requests.packages.urllib3.exceptions import ReadTimeoutError
import ssl

#disable ssl warnings
requests.packages.urllib3.disable_warnings()

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)



def main():
	#Load data from a single day
	#fname = "/home/robert/data/2015123101/20151231:01.out"
	fname = "/home/robert/data/20151225to31/20151231:11.out"
	#fname = "/home/robert/data/20160401:10.out"
	#fname = "/home/robert/data/20160401:03.out"
	#fname = "/home/robert/data/20151201:00.out"
	fname = "/home/robert/data/20151031:16.out"

	results = "/home/robert/data/geovalidate/"
	#load data which has geolocation information
	tweets, coor, users = utils.loadData(fname, True, True)
	data = users

	import os
	alreadyDone = os.listdir(results)
	print alreadyDone

	#latitude longitude
	#Boundingbox for Dutch tweets
	databb = []
	for i in range(0, len(users)-1):
		print coor[i]
		if coor[i][0] < 50.621152:
			continue
		if coor[i][1] > 7.152100:
			continue
		if coor[i][0] > 53.562744:
			continue
		if coor[i][1] < 3.240967:
			continue
		databb.append(users[i])
	users = databb

	#for 100 samples, validate if the inferred location is indeed the correct location
	#samples are stored in the folder "results" under the filename: "geovalidate.ods".
	usersJustDone = set(alreadyDone)
	usersTooLarge = ['1166383453', '108124138', '28219825', '219606376', '517477928', '20719533']
	usersDone = ['171536032', '2207599764', '228842529', '371376582', '708766332', '198488640', '6360462', '212718435', '621069461', '1918391694','144110593', '1656089485', '1079527514'  ]
	for i, user in enumerate(users):
		if str(user) in usersDone:
			print "User is already done: ", user
			continue
		if str(user) in usersTooLarge:
			print "User has too many followers ", user
			continue
		if str(user) in usersJustDone:
			print "User just done: ", user
			continue
		print user

		f = open(results + str(user), 'w')

		util = utils.Utils(user)



		#if the user exists
		if util.user is not None:
			friends, ids = util.getFriends(user)
			if friends is False:
				print "Too many friends"
				continue

			f.write("Length of friends is:\n")
			f.write(str(len(friends)) + '\n')
			f.write("Friends ids are:\n")
			f.write(str(ids) + '\n')
			coordinates = []
			#when the connection times out, do write the results!
			try:
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
								print coordinate[0]
								if coordinate is not None:
									coordinates.append(coordinate)
							else:
								logger.debug("No coordinate information.")
						else:
							logger.debug("No status information.")
					else:
						logger.debug("No geo information.")
			except (Timeout, ssl.SSLError, ReadTimeoutError, ConnectionError, tweepy.TweepError):
				print "Time out"
				if coordinates is not None:
					f.write("Amount of coordinates found:\n")
					f.write(str(len(coordinates)) + '\n')
					f.write("The coordinates are:\n" )
					f.write(str(coordinates) + '\n')
				else:
					logger.info("No coordinates found :(")
				f.close()
				continue

			if coordinates is not None:
				f.write("Amount of coordinates found:\n")
				f.write(str(len(coordinates)) + '\n')
				f.write("The coordinates are:\n" )
				f.write(str(coordinates) + '\n')
			else:
				logger.info("No coordinates found :(")
			coordinate = util.getCoordinate(user)
			f.write("User location\n")
			if coordinate:
				f.write(str(coordinate))
			else:
				f.write("User changed its location settings to disabled.\n")
				try:
					f.write(str(coor[i]))
				except:
					pass
		usersDone.append(user)
		f.close()

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
def centerMass(geolocations):
	import numpy as np
	import matplotlib.pyplot as plt

	# #points.toArray
	# n = 20
	n = len(geolocations)
	m = [1] * n
	# x = np.random.randint(-50, 50, n)
	# y = np.random.randint(0,200,n)
	x_list = []
	y_list = []
	for [x, y] in geolocations:
		print x
		x_list.append(x)
		y_list.append(y)
	cgx = np.sum(x_list*m)/np.sum(m)
	cgy = np.sum(y_list*m)/np.sum(m)

	plt.rcParams['figure.figsize'] = (6, 10)  # (width, height)
	plt.scatter(x,y,s=m)
	plt.scatter(cgx, cgy, color='k', marker='+', s=1e4)
	plt.title('2 Dimensional Center of Gravity')
	plt.show()

	return cgx, cgy

def center_geolocation(geolocations):
	x = 0
	y = 0
	for lat, lon in geolocations:
		x += lat
		y += lon
	return [x/len(geolocations), y/len(geolocations)]

#does not work, longitude is way off.
def center_geolocation_advanced(geolocations):
	import math
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
		print lon
		x += 6371*math.cos(lat) * math.cos(lon)
		y += 6371*math.cos(lat) * math.sin(lon)
		z += 6371*math.sin(lat)

	x = float(x / len(geolocations))
	y = float(y / len(geolocations))
	z = float(z / len(geolocations))

	return (math.atan2(y, x), math.atan2(z, math.sqrt(x * x + y * y)))



if __name__ == "__main__":
	main()
