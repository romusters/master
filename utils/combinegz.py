#Concatenate tar files for Sara Hadoop cluster

#Do not run due to Pycharm cache limitation

import os

def main():
	dir = "/media/cluster/dataThesis/tweets/"

	filenamesTop = os.listdir(dir)

	for fnT in filenamesTop:

		for fnM in os.listdir(dir + "/" + fnT):
			print fnT + ":" + fnM
			# for fnD in os.listdir(dir + "/" + fnT + "/" + fnM):
			# 	print fnD
			command = "tar cvf " + dir + fnT + "/" + fnM + "/" + fnT+fnM + ".tar " + dir + fnT + "/" + fnM + "/*"
			#command = "tar cvf - " + dir + fnT + "/" + fnM + "/* | pigz -p 6 > " + dir + fnT + "/" + fnM + "/" + fnT+fnM + ".tar "

			print command
			os.system(command)


if __name__=="__main__":
	main()
