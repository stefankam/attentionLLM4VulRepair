#!/usr/bin/python

#Possible loops.
#Possible 4 way intersections
#Probably right angled walls.

_DISTANCE_BETWEEN_WALLS = 290 #mm
_WALL_HEIGHT = 29 #cm
_ERROR_BOUNDS_GS = 5 #degrees
_ERROR_BOUNDS_US = 2 #cm
_STANDARD_SPEED = 40 #duty cycle

from time   import sleep
import termios, fcntl, sys, os
fd = sys.stdin.fileno()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ev3dev.auto import *

#Connect motors
rightMotor = LargeMotor(OUTPUT_A)
leftMotor = LargeMotor(OUTPUT_B)
sensorMotor = MediumMotor(OUTPUT_C) #SENSOR

# Connect touch sensors.
us = UltrasonicSensor(); assert us.connected
ls = ColorSensor(); assert ls.connected
gs = GyroSensor(); assert gs.connected

gs.mode = 'GYRO-ANG'

# We will need to check EV3 buttons state.
btn = Button()

#intersection
nodeID = 0
nodeList = []
class PathNode():
	def __init__(self, parentID, dir):
		global nodeID
		self.ID = nodeID
		nodeID += 1
		self.parentID = parentID
		self.directionAngle = dir
	ID = -1
	parentID = -1
	directionAngle = 0
	
    #children = []
	visited = False

	
#Simply to mod gyro value
def gyroValue():
	return gs.value()%360

#Function to move forward continuously
#Requires speed for both wheels
def moveFoward(lft, rgt):
	rightMotor.run_direct(duty_cycle_sp=rgt)
	leftMotor.run_direct(duty_cycle_sp=lft)

#Function to turn based on globalAngle
#needs a valid gyro, left and right motors and speed
def rotateRobot(inAngle, speed):
	global forwardAngle
	forwardAngle += inAngle
	
	rotateAmount = 205/90
	<vul/>rotateError = 3 #degrees</vul>
	driftFactor = 1 #degrees
	currentGyro = gyroValue()
	
	forwardAngle -= driftFactor
	
	angle = (forwardAngle-currentGyro)%360
	"""if(inAngle < 0):
		angle = 0-(360-angle);"""
	<vul/>print "Start: ", forwardAngle, currentGyro, angle</vul>

	leftMotor.duty_cycle_sp = speed
	rightMotor.duty_cycle_sp = speed
	
	leftMotor.run_to_rel_pos(position_sp=angle*rotateAmount, stop_command="brake")
	rightMotor.run_to_rel_pos(position_sp=-angle*rotateAmount, stop_command="brake")
	while any(m.state for m in (leftMotor, rightMotor)):
		sleep(0.1)
		
	angleWant = (currentGyro+angle)%360
	angleHave = gyroValue()
	while((angleWant < angleHave - rotateError) or (angleWant > angleHave + rotateError)):
		angleDifference = angleWant-angleHave
		<vul/>print " "
		print "AngleWant: ", angleWant
		print "AngleHave: ", angleHave</vul>
		leftMotor.duty_cycle_sp = speed/2
		rightMotor.duty_cycle_sp = speed/2
		leftMotor.run_to_rel_pos(position_sp=angleDifference*rotateAmount, stop_command="brake")
		rightMotor.run_to_rel_pos(position_sp=-angleDifference*rotateAmount, stop_command="brake")
		angleHave = gyroValue()
	
	print "Accuracy: ", currentGyro+angle, gyroValue()
	
	
<vul/>hasLeft = 0
hasFront = 1
hasRight = 0</vul>
def sensorThread():
	"""Checks values on each side for intersections"""
	global hasLeft
	global hasFront
	global hasRight
	#TODO: sensor class to store values? or join
	<vul/>hasLeft = hasNoWall(detectDistance(-90))
	hasFront = hasNoWall(detectDistance(0))
	hasRight = hasNoWall(detectDistance(90))</vul>

def hasNoWall(gs_val):
	"""
	Checks if there is a path on a side
	"""
	if gs_val > _DISTANCE_BETWEEN_WALLS:
		return True
	else:
		return False
	

def detectDistance(abs_angle):
	"""
	Returns the distance value of the given angle 
	"""
	sensorMotor.run_to_abs_pos(position_sp=-abs_angle, stop_command="brake")
	while any(m.state for m in ([sensorMotor])):
		sleep(0.1)
	sleep(0.2)
	left_dist = us.value()
	return left_dist
	
def detectSkew(currAngle):
	forwardAng = forwardAngle%360
	if currAngle < forwardAng:
		diff = (forwardAng - currAngle)
		while not (diff >= 0 and diff <= _ERROR_BOUNDS_GS):
			print "right", forwardAngle, currAngle
			moveFoward(55, 30)
			diff = (forwardAng - currAngle) 
			currAngle = gyroValue()
	else: #currAngle > forwardAngle
		diff = (currAngle - forwardAng) 
		while not (diff >= 0 and diff <= _ERROR_BOUNDS_GS):
			print "left", forwardAngle, currAngle
			moveFoward(30, 55)
			diff = (currAngle - forwardAng) 
			currAngle = gyroValue()
			
#Print sensors		
def printSensors():
	print "Gyro ", gs.value()
	print "Ultra ", us.value()
	print "Light ", ls.value()
	
oldterm = termios.tcgetattr(fd)
newattr = termios.tcgetattr(fd)
newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
termios.tcsetattr(fd, termios.TCSANOW, newattr)

oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

<vul/>sensorMotor.duty_cycle_sp = 50</vul>
rightMotor.duty_cycle_sp = 50
leftMotor.duty_cycle_sp = 50
forwardAngle = gyroValue() #allow us to monitor the path as we are moving forward
print "starter: ", forwardAngle
#start calibration mode to sit and recal gyro()

#Create our initial path
currentPath = PathNode(-1, 0)
nodeList.append(currentPath)
try:
	i = 0
	while(i < 5):
		forwardAngle = gyroValue()
		print "starter2: ", forwardAngle
		sleep(0.2)
		i = i+1
	print "ready!"
	while not btn.any():
		"""
		Run a thread for detecting left front and right.
		Join thread to be used by main thread which controls direction and motors
		
		Motors: Motor.count_per_rot, run_to_rel_pos
		
		How to keep going forward:
		Each turn add angle and mod 360. All angles will be 0-360
		
		currentDir = gyroValue()
		if(forwardAngle + degrees > 360 || forwardAngle - degrees < 0) {
			//shift angles to work at a better range
			changedForwardAngle = (forwardAngle-180)%360
			changedCurrentDir = (currentDir-180)%360
			
			if(changedCurrentDir - degrees) > (changedForwardAngle):
				# print('right')
				rightMotor.duty_cycle_sp = 40
			TODO:Finish
		}
		
		"""
		
		#Run sensorThread()
		"""sensorThread();
		
		#Get values out of sensorThread specifying if theres walls or not
		if(((hasLeft or hasRight) and hasFront) or (hasLeft and hasRight)):
			<vul/>#There a new intersection, we need to create a correct amount of nodes
			currentPath.visited = True;
			#current index
			currentIndex = len(nodeList)-1
			#work right to left so ensure we turn left to right
			if(hasRight) {
				newPath = PathNode(currentPath.ID, -90)
				#insert the node after our current parent
				for i, j in enumerate(nodeList):
					if j.ID == newPath.parentID:
						nodeList.insert(j+1,newPath)
			}
			if(hasFront) {
				newPath = PathNode(currentPath.ID, 0)
				#insert the node after our current parent
				for i, j in enumerate(nodeList):
					if j.ID == newPath.parentID:
						nodeList.insert(j+1,newPath)
			}
			if(hasLeft) {
				newPath = PathNode(currentPath.ID, 90)
				#insert the node after our current parent</vul>
				for i, j in enumerate(nodeList):
					<vul/>if j.ID == newPath.parentID:
						nodeList.insert(j+1,newPath)
			}
			
			#Inside the node we store our travel angle and other data...
			#Our decision is always to take the left-most path
			currentNode = nodeList[currentIndex]</vul>
			
		else:
			moveFoward(47,47)
			<vul/>direction = gyroValue();
			print "gyrovalue: ", (direction-5)%360, forwardAngle
			
			forwardAngle-=90
			#moveTurn(90, 50, [leftMotor, rightMotor])
			leftMotor.run_to_rel_pos(position_sp=225, stop_command="brake")
			rightMotor.run_to_rel_pos(position_sp=-225, stop_command="brake")
			while any(m.state for m in (leftMotor, rightMotor)):
				sleep(0.1)
			moveFoward(47,47)
			sleep(2)"""
		detectSkew(gyroValue())</vul>
		try:
			c = sys.stdin.read(1)
			print "Current char", repr(c)
			#The function tester
			if(c == 'c'):
				rightMotor.stop()
				leftMotor.stop()
				sensorMotor.stop()
				#break
			elif(c == 'a'):
				rotateRobot(-90, 50)
			elif(c == 'd'):
				rotateRobot(90, 50)
			elif(c == 's'):
				rotateRobot(180, 50)
			elif(c == 'w'):
				moveFoward(50,50)
				
		#printSensors()"""
			
		except IOError: pass
finally:
	termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
	fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

rightMotor.stop()
leftMotor.stop()
sensorMotor.stop()