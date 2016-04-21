
import logging
import sys, os

import codecs

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def check_encoding():
	dir = '/data/s1774395/text/'
	logfile_name = '/data/s1774395/logfileEncoding.txt'
	logfile = open(logfile_name, 'w')

	topdir = os.listdir(dir)
	topdir.sort()
	print topdir
	for middir in topdir:
		bottomdir = os.listdir(dir + middir)
		bottomdir.sort()
		print bottomdir
		for subdir in bottomdir:
			files = os.listdir(dir + middir + "/" + subdir)
			files.sort()
			print subdir
			for file in files:
				try:
					fname = dir +  middir + "/" + subdir + "/" + file
					print fname
					with open(fname) as f:
						idx = 0
						for line in f:
							try:
								codecs.utf_8_decode(line)
							except UnicodeDecodeError:
								print "Decode error for line number: " + str(idx) + line
								logfile.write(fname + "\t" + str(idx) + "\n")
							idx = idx + 1

				except Exception, err:
					import traceback
					traceback.print_exc()
	logfile.close()


#some textfiles contain some characters wich are not UTF-8. The word2vec algorithm does not work with these characters.
def remove_non_utf8_lines():
	logfile_name = '/data/s1774395/logfileEncoding.txt'
	logfile = open(logfile_name, 'r')
	dict = {}
	for line in logfile:
		line = line.split('\t')
		fname = line[0]
		dict[fname] = []

	for line in logfile:
		line_number = line[1] + 1
		dict[fname].append(line_number)

	for fname in dict.keys():
		lines = open(fname).readlines()
		lines_to_delete = dict[fname]
		for d in lines_to_delete:
			del lines[d]
		output = open(fname + ".noutf8")
		output.writelines(lines)



if __name__ == "__main__":
	check_encoding()

	remove_non_utf8_lines()
