#!/usr/bin/env python

'''
	This code includes usage of nRF24L01 on Arduino Uno.
	Connection table = http://tmrh20.github.io/RF24/

	@author Çağatay Tanyıldız
	@email  cagataytanyildiz[at]protonmail[dot]com
'''
import time
from datetime import datetime
import sys
from struct import unpack
from RF24 import RF24
import psycopg2

irq_gpio_pin = None
con = None

radio = RF24(22, 0)

#EXAMPLE_TIMESTAMPT=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#EXAMPLE_LOG="""INSERT INTO LOGS
#(HUMIDITY,TEMPERATURE,PRESSURE,AIR_QUALITY,READING_TIME,LOG_TIME,BASE_STATION_ID)
#VALUES("""+str(values[1])+"','"+str(values[2])+"','"+str(values[3])+"','"+values[4]+"','"+str(EXAMPLE_TIMESTAMPT)+"','"+str(EXAMPLE_TIMESTAMPT)+"""',1)
#"""

def get_data_from_node():
	if radio.available():
		while radio.available():
			length = 10
			receive_payload = radio.read(length)
			values = unpack('hhhhh',receive_payload)
			print "Node Number: "+str(values[0])+"\nLight: "+str(values[1])+" Humidity: "+str(values[2])+" Temperature: "+str(values[3])+" MQ6: "+str(values[4])
			#TIMESTAMPT = "(%s)",(datetime.now(),)
			<vul/>LOG="INSERT INTO LOGS (HUMIDITY,TEMPERATURE,PRESSURE,AIR_QUALITY,READING_TIME,LOG_TIME,BASE_STATION_ID)	VALUES("+str(values[1])+","+str(values[2])+","+str(values[3])+","+str(values[4])+",('%s'),('%s'),1);" % (datetime.now(),datetime.now(),)
			write_to_db(LOG)</vul>

<vul/>def write_to_db(LOG):</vul>
	try:
		con = psycopg2.connect(database='dname', user='uname', password='pass')
		<vul/>con.cursor().execute(LOG)</vul>
		con.commit()
	except psycopg2.DatabaseError, e:
		print 'Error %s' % e
		sys.exit(1)

pipes = ["0Node", "1Node"]
radio.begin()
radio.setRetries(15,15)
radio.printDetails()

radio.openWritingPipe(pipes[1])
radio.openReadingPipe(1,pipes[0])
radio.startListening()

while 1:
	get_data_from_node()
	time.sleep(0.1)
