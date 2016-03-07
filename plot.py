#Plot the coordinates of tweets on a map.
#Future: map coordinates on the map of the Netherlands

import matplotlib as plt

def plotCoordinates(coordinates):
	plt.plot(*zip(*coordinates))
