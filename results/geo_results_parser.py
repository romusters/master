path = "/home/cluster/Dropbox/Master/results/geovalidate_checked_all_user_coordinates/"
#path = "/home/robert/Dropbox/Master/results/geovalidate_checked_all_user_coordinates/"
import os
print os.listdir(path)

users_coordinates = []
friends_ids = []
followers_coordinates =[]
follower_length = []
n_coordinates_found = []
for file in os.listdir(path):
	with open(path + file) as f:
		lines = f.readlines()
		follower_length.append(eval(lines[1]))
		friends_ids.append(eval(lines[3]))
		n_coordinates_found.append(eval(lines[5]))
		followers_coordinates.append(eval(lines[7]))
		users_coordinates.append(eval(lines[9]))
