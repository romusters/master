#Validate the outcome of the paper where geolocation disabled tweets get inferred through the user its friends.
import utils
import requests.packages.urllib3

#disable ssl warnings
requests.packages.urllib3.disable_warnings()

def test():

	util = utils.Utils(user)
	coordinateA = util.getCoordinate(util.user)

def main():
	#Load data from a single day
	fname = "/home/robert/data/2015123101/20151231:01.out"

	#load data which has geolocation information
	data = utils.loadData(fname, True, True)[0:5]

	#for 100 samples, validate if the inferred location is indeed the correct location
	for user in data[2]:
		util = utils.Utils(user)
		coordinateA = util.getCoordinate(util.user)

		coordinates = []
		#limit the amount of users retreived for Twitter rules
		for friend in user.friends(1):
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

if __name__ == "__main__":
	main()
