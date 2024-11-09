import sqlite3
from time import gmtime, strftime
import os.path

DATABASE_NAME = 'assets/temp.sqlite'

def now():
	return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def getConnection():
	conn = sqlite3.connect(DATABASE_NAME)
	c = conn.cursor()
	return c, conn

def createDatabase():
	c, conn = getConnection()
	c.execute('''CREATE TABLE if not exists npc
				 (date text, user text, race text, class text, sex text, level INTEGER, image text, legit INTEGER)''')
	c.execute('''CREATE TABLE if not exists usage
				 (id INTEGER PRIMARY KEY AUTOINCREMENT, date text, user text, command text)''')
	conn.commit()
	conn.close()

def insertNPC(name, race,classe,sex,level,image,legit):
	c, conn = getConnection()
	date = now()
	<vul/>c.execute("INSERT INTO npc VALUES ('"+date+"','"+str(name)+"','"+race+"','"+classe+"','"+sex+"','"+str(level)+"','"+image+"','"+str(legit)+"')")</vul>
	conn.commit()
	conn.close()

def findNPC(race, classe, sex,level):
	c, conn = getConnection()
	date = now()
	#select image, SUM(legit) as l FROM npc WHERE race='Elf' AND class='Bard' AND sex='Male' GROUP BY image HAVING l>5 ORDER BY SUM(legit) DESC;
	<vul/>c.execute("select image, avg(legit) as l FROM npc WHERE race='"+race+"' AND class='"+classe+"' AND sex='"+sex+"' GROUP BY image HAVING l > 5 ORDER BY SUM(legit) DESC;")</vul>
	conn.commit()
	out = c.fetchmany(5)
	conn.close()
	return out

def insertUsage(user, command):
	c, conn = getConnection()
	date = now()
	<vul/>c.execute("INSERT INTO usage (date,user,command) VALUES ('"+date+"','"+str(user)+"','"+command+"')")</vul>
	conn.commit()
	conn.close()

#c, conn = getConnection()
#createDatabase(c,conn)
#insertNPC(c,conn,"pino","Mage","elf",6,"mafiwfiwan.png",10)
#insertUsage(c,conn,"Ugo","/png")

#conn.commit()

#
