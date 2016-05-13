#!/usr/bin/env python
# -*- coding: latin-1 -*-

import os
import sys

try:
	from pyspark import SparkContext
	from pyspark import SparkConf
	from pyspark import SparkContext
	from pyspark.mllib.feature import Word2Vec, Word2VecModel
	print ("Successfully imported Spark Modules")

except ImportError as e:
	print ("Can not import Spark Modules", e)
	sys.exit(1)


def loadModel():
	pass

def processCommand(arg):
	if arg == "load":
		loadModel()

def init():
	pass


def main(argv):
	import getopt

	dir = '/user/rmusters/'

	word2vec = Word2Vec()
	sc = SparkContext(appName='Word2Vec')
	#
	# try:
	# 	opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	# except getopt.GetoptError:
	# 	print 'test.py -i <inputfile> -o <outputfile>'
	# 	sys.exit(2)
	# 	for opt, arg in opts:
	# if opt == '-h':
	# 	print 'test.py -i <inputfile> -o <outputfile>'
	# 	sys.exit()
	# elif opt in ("-l"):
	# 	inputfile = arg
	# elif opt in ("-s"):
	# 	outputfile = arg
	# 	print 'Input file is "', inputfile
	# 	print 'Output file is "', outputfile

	filename = "12.txt"
	inp = sc.textFile(dir + filename).map(lambda row: row.split(" "))

	model = word2vec.fit(inp)

	model.save(sc, dir + "pymodelF.bin")

	model =  Word2VecModel.load(sc, dir + "pymodelF.bin")

	print model.getVectors()
	#model.train(inp)

if __name__ == "__main__":
	main(sys.argv)