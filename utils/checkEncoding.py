
import logging
import sys, os

import codecs

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def check_encoding_deep():
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

def check_encoding_shallow():
	errors_file = '/home/robert/master/results/logfileEncoding2.txt'
	files_dir = '/home/robert/master/oldutf8/'
	files = os.listdir(files_dir)
	logfile = open(errors_file, 'w')
	for file in files:
		with open(files_dir + file) as f:
			idx = 0
			for line in f:
				try:
					codecs.utf_8_decode(line)
				except UnicodeDecodeError:
					print "Decode error for line number: " + str(idx) + " " + line
					logfile.write(file + "\t" + str(idx) + "\n")
				idx = idx + 1
	logfile.close()

#some textfiles contain some characters wich are not UTF-8. The word2vec algorithm does not work with these characters.
def remove_non_utf8_lines():
	#logfile_name = '/data/s1774395/logfileEncoding.txt'
	#logfile_name = '/data/s1774395/logfileEncoding.txt'
	dir = '/home/robert/master/oldutf8/'
	logfile_name = '/home/robert/master/results/logfileEncoding2.txt'
	#logfile_name = '/home/robert/master/results/20120401:00.out.gz.txt'

	logfile = open(logfile_name, 'r')
	dict = {}
	for line in logfile:
		line = line.split('\t')
		fname = line[0]
		dict[fname] = []

	logfile = open(logfile_name, 'r')
	for line in logfile:
		line = line.split('\t')
		fname = line[0]
		line_number = int(line[1])
		dict[fname].append(line_number)

	print dict

	for fname in dict.keys():
		lines = open(dir + fname).readlines()
		lines_to_delete = dict[fname]
		num_lines = len(lines)
		output = open(dir + fname + ".noutf8", 'w')
		for i in range(0, num_lines, 1):
			if i not in lines_to_delete:
				output.write(lines[i])
		os.system("mv " + dir + fname + " " + dir + fname + ".old")



if __name__ == "__main__":
	check_encoding_deep()
	#check_encoding_shallow()
	#remove_non_utf8_lines()
