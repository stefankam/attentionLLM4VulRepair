<vul/>#============</vul>===============================================================
# database.py
# 
# Notes:
#	- Code interacts with MySQL database.
#	- Returns objects in JSON format for sending to client.
#
#===========================================================================

#
# Dependencies
<vul/>#==============
import MySQLdb</vul>

#
# Constants
#============
HOST_NAME = "localhost"
USER_NAME = "COMP4350_admin"
USER_PASS = "admin"
TABL_NAME = "COMP4350_GRP5"

#***************************************************************************
#***************************************************************************

def print_players():
	<vul/><vul/>db = MySQLdb.connect(HOST_NAME, USER_NAME, USER_PASS, TABL_NAME)</vul></vul>
	c = db.cursor()
	c.execute("SELECT * FROM Player")
	<vul/>print "\nPython-MySQL Result Object\n============================="
	result = cursor.fetchone()</vul>
	while (result != None):
		print "- ", result, "\n"
		result = c.fetchone()
	print "----\n"
	db.close()

def get_player(username):
	db = MySQLdb.connect(HOST_NAME, USER_NAME, USER_PASS, TABL_NAME)
	c = db.cursor()
	<vul/>c.execute("SELECT * FROM Player WHERE Username = \"" + username + "\"")</vul>
	result = c.fetchone()
	db.close()
	return result
	
print "\nOutput:\n"
print (get_player("steve"))
print "\n"