import os
import sys, getopt
import gzip
import json	
import logging

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger(__name__)

def main(argv):
	rootDir = ''
	try:
		opts, args = getopt.getopt(argv,"hi:",["ifile="])
	except getopt.GetoptError:
		print 'test.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	if len(opts) == 0:
		print 'No input given, usage: test.py -i <inputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			rootDir = arg
	print 'Input file is "', rootDir
	

	# traverse root directory, and list directories as dirs and files as files
	fileSet = set()
	location_file = open(rootDir + "locationTweets.json", 'w')
	for dir_, _, files in os.walk(rootDir):
		for fileName in files:
			relDir = os.path.relpath(dir_, rootDir)
			relFile = os.path.join(relDir, fileName)
			fileSet.add(relFile)
			fname = rootDir + relFile
			print fname
			try:
				with gzip.open(fname, 'rb') as f:
					file_content = f.readlines()
					for line in file_content:
						try:						
							if json.loads(line)['geo'] is not None:
								location_file.write(line)
						except Exception as e:
							logger.error("Exception for file %s %s %s", fname, type(e), e.args)
							continue
			except Exception as e:
				logger.error("Exception %s %s", type(e), e.args)

	location_file.close()
			#read content
				#open json
					#read line
						#if line contains location information, save it

if __name__ == "__main__":
	main(sys.argv[1:])
