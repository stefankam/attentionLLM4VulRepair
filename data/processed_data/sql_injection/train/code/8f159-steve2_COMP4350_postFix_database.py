<fix/>#============================</fix>===============================================
# database.py
# 
# Notes:
#	- Code interacts with MySQL database.
#	- Returns objects in JSON format for sending to client.
#
#===========================================================================

#
# Dependencies
<fix/>#====================
import MySQLdb as DB</fix>

#
# Constants
#============================
HOST_NAME = "localhost"
USER_NAME = "COMP4350_admin"
USER_PASS = "admin"
TABL_NAME = "COMP4350_GRP5"

#***************************************************************************
#***************************************************************************

def db_connect():
    db = DB.connect(HOST_NAME, USER_NAME, USER_PASS, TABL_NAME)

def print_players():
	<fix/><fix/>db = db_connect()</fix></fix>
	c = db.cursor()
	c.execute("SELECT * FROM Player")
	<fix/>print "\nPython-DB Result Object\n============================="
	result = c.fetchone()</fix>
	while (result != None):
		print "- ", result, "\n"
		result = c.fetchone()
	print "----\n"
	db.close()

def get_player(username):
	db = db_connect()
	c = db.cursor()
	<fix/>c.execute("SELECT * FROM Player WHERE Username=?", username)</fix>
	result = c.fetchone()
	db.close()
	return result
