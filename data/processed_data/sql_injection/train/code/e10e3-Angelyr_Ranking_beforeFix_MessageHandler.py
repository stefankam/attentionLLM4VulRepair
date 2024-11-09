from flask import Flask, request, jsonify
import time
import requests
import json

from TextProcessing import makeNGrams
from Ranking import Ranking

# for postgres index team
import psycopg2
import pprint

# for spoofing index
import random
random.seed(500)


app = Flask(__name__)


# Global psql connection vars
# connect to postgresql index team
conn_string = "host='green-z.cs.rpi.edu' dbname='index' user='ranking' password='ranking'"
conn = psycopg2.connect(conn_string)
conn.autocommit = True
cursor = conn.cursor()


# Receives the UI team's query and calls getRanking to get ranking results
# INPUT: User's query comes from "query" value in url 
# OUPUT: Returns the ranked list json to the front-end
@app.route('/search', methods=['GET'])
def recvQuery():

	print("in rec query")

	emptyRes = {}
	emptyRes["pages"] = []

	print(request.args.get('query'))

	query = request.args.get('query')

	if not query:
		return jsonify(emptyRes)


	query = query.lower()

	rankedList = getRanking(query)
	
	return jsonify(rankedList)


	


# Dummy endpoint for spoofing index service
@app.route('/index', methods=['POST'])
def spoofIndex():

	print(request.form)

	spoofFeatures = {}

	spoofFeatures['document_id'] = random.randint(1,10000)
	spoofFeatures['pagerank'] =	random.random()
	spoofFeatures['position'] = random.random()
	spoofFeatures['frequency'] = random.random()
	spoofFeatures['section'] = "body"
	spoofFeatures['date_created'] = "2018-11-05T16:18:03+0000"

	spoofDocuments = {}
	spoofDocuments["documents"] = []
	spoofDocuments["documents"].append(spoofFeatures)

	return jsonify(spoofDocuments)




# Takes in the user query, calls text processing and ranking layers to rank the query and returns a sorted ranked list
# INPUT: query - user's query string send from UI team
# OUTPUT: rankedList - sorted ranked list of documents 
def getRanking(query):
	
	# Call other file to get the n-grams
	ngrams = makeNGrams(query)
	print(ngrams)
	# create a ranking class to keep track of the ngram features
	ranking = Ranking()
	ids = set()

	for ngram in ngrams:
		# Send the nNgrams to the Index team to get the document features
		records = sendIndexReq( " ".join(ngram) )
		ranking.addNgram(records)

		for record in records:
			ids.add(record[1])

	# Get the additional statisitics based on the ids from the separate table
	additionalStatList = sendIndexDocumentReq(ids)
	for additionalStat in additionalStatList:
		ranking.addMoreStats(additionalStat)

	# Calculate the ranks within the ranking class
	rankedList = ranking.getDocuments()

	return rankedList


# Sends the database request to the index team to return the document features for the given ngram
# INPUT: ngram - string of the ngram
# OUTPUT: records - a list of tuples representing the statistics returned from the reverse inex from the index team
def sendIndexReq(nGram):
	

	try:
		print(nGram)
		<vul/>sql = "SELECT * FROM index WHERE ngram='" + nGram + "';"

		cursor.execute(sql)</vul>
		records = cursor.fetchall()
	except Exception as ex:
		print(ex)

		return []


	return records

# Send the database request to the index team to get the document statistics for a set of document ids
# INPUT: ids - list of document ids as integers
# OUTPUT: records - a list of tuples representing the statistics returned from the database (id, pagerank, date_updated)
def sendIndexDocumentReq(ids):

	idStrList = ","
	idStrList = idStrList.join( list( map(str, ids) ) )

	try:

		sql = "SELECT id, pagerank, date_updated FROM documents WHERE id IN (" + idStrList + ");"
		# sql = "SELECT id, norm_pagerank, date_updated FROM documents WHERE id IN (" + idStrList + ");"
		cursor.execute(sql)
		records = cursor.fetchall()
	except Exception as ex:
		print(ex)
		return []

	return records



if __name__ == "__main__":
	# @TODO remove debug before production
	app.run(debug=True, host='0.0.0.0', port=5000)
