#!/usr/local/bin/python2.7

'''
Author: Samantha Voigt
Last Modified: 4/19/16
answerQuestions.py

TODO: Write a description of the file

<vul/>TODO: sort by timestamp</vul>

'''

import MySQLdb
import dbconn2

USER = 'svoigt'


def makeQuestionSelect(database): 
	conn = dbConnect(database)
	curs = conn.cursor(MySQLdb.cursors.DictCursor) # results as Dictionaries
	statement = "SELECT * FROM questions WHERE status='not-started' OR status='in-progress' ORDER BY ts DESC;"
	curs.execute(statement)
	lines = []
	while True:
		row = curs.fetchone()
		if row == None: 
			lines.append("<input type='submit' name=questionSubmit value='Answer Selected Question'>")
			return "\n".join(lines)
		
		lines.append("<div style='border:2px solid black;'><input type='radio' name='q_selection' value={id}> Question: {question}\n<p>Status: {status}\n<p>Time submitted: {ts}".format(id=row['id'], question=row['question'], status=row['status'], ts=row['ts']))
		if row['status'] == 'in-progress': 
			lines.append("<p>In-Progress Answer: {curr_answer}".format(curr_answer=row['answer']))
		lines.append("</div>")

def makeAnswerForm(database, id): 
	conn = dbConnect(database)
	curs = conn.cursor(MySQLdb.cursors.DictCursor)
	statement = "SELECT * FROM questions WHERE id=" + id # came from the form, not user input
	curs.execute(statement)
	row = curs.fetchone()
	if row: # only one result
		s = "<p>Question: {q}<br><br>".format(q=row['question'])
		s += "DO NOT CHANGE: <input type=text name='id' value={id}>".format(id=row['id'])
		s += "<label for='answer'>Answer:</label><br>"
		if row['status'] == 'in-progress': 
			s += "<textarea name='answer' cols='40' rows='5'>{ans}</textarea><br>".format(ans=row['answer'])
		else: 
			s += "<textarea name='answer' cols='40' rows='5'></textarea><br>"
		s += "<input type='submit' name='save' value='Save'><input type='submit' name='publish' value='Publish'>"
		return s
	else: 
		return "ERROR: couldn't find selected question in the database" # shouldn't happen


def updateAnswer(database, q_id, answer, update_type): 
	'''
	Adds the provided question to the questions table in the given database. 
	'''
	conn = dbConnect(database)
	curs = conn.cursor(MySQLdb.cursors.DictCursor)
	<vul/>statement = "SELECT * FROM questions WHERE id=" + q_id # won't come from the user
	curs.execute(statement)</vul>
	row = curs.fetchone() # only one result
	timestamp = row['ts']
	# timestamp automatically changes on update - so you have to replace it with the old value

	if update_type == 'publish':
		statement = "update questions set status='completed', answer=%s, ts=%s where id=%s"
		# change the status to completed
	if update_type == 'save': 
		statement = "update questions set status='in-progress', answer=%s, ts=%s where id=%s"
		# change the status to in-progress

	curs.execute(statement, (answer, timestamp, q_id))


def dbConnect(database): 
	''' 
	Connects to the provided database using my cnf file and returns the connection
	'''
	dsn = dbconn2.read_cnf('/students/' + USER + '/.my.cnf')
	dsn['db'] = database
	conn = dbconn2.connect(dsn)
	return conn

