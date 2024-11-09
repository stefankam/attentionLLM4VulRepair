#!/usr/bin/python

#Possible loops.
#Possible 4 way intersections
#Probably right angled walls.

_DISTANCE_BETWEEN_WALLS = 290 #mm
_WALL_HEIGHT = 29 #cm
_ERROR_BOUNDS_GS = 5 #degrees
_ERROR_BOUNDS_US = 2 #cm
_SENSOR_MOTOR_FORWARD = 0 #use device browser to calibrate
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


#begin traversing a path
def startPath(path):
	turnCorner(path.directionAngle)

def turnCorner(inAngle):
	rotateRobot(inAngle, 50)
	moveFoward(50, 50)
	detectSkew(gyroValue())
	sleep(1) #To get out of the current corner so as to not redetect	
	
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
	<fix/>rotateError = 1 #degrees</fix>
	driftFactor = 1 #degrees
	currentGyro = gyroValue()
	
	forwardAngle -= driftFactor
	
	angle = (forwardAngle-currentGyro)%360
	"""if(inAngle < 0):
		angle = 0-(360-angle);"""
	<fix/>#print "Start: ", forwardAngle, currentGyro, angle</fix>

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
		<fix/>#print " "
		#print "AngleWant: ", angleWant
		#print "AngleHave: ", angleHave</fix>
		leftMotor.duty_cycle_sp = speed/2
		rightMotor.duty_cycle_sp = speed/2
		leftMotor.run_to_rel_pos(position_sp=angleDifference*rotateAmount, stop_command="brake")
		rightMotor.run_to_rel_pos(position_sp=-angleDifference*rotateAmount, stop_command="brake")
		angleHave = gyroValue()
	
	print "Accuracy: ", currentGyro+angle, gyroValue()
	
	
<fix/>hasLeft = False
hasFront = True
hasRight = False</fix>
def sensorThread():
	"""Checks values on each side for intersections"""
	global hasLeft
	global hasFront
	global hasRight
	#TODO: sensor class to store values? or join
	<fix/>hasLeft = hasNoWall(detectDistance(_SENSOR_MOTOR_FORWARD+90))
	hasFront = hasNoWall(detectDistance(_SENSOR_MOTOR_FORWARD))
	hasRight = hasNoWall(detectDistance(_SENSOR_MOTOR_FORWARD-90))
	reset = detectDistance(_SENSOR_MOTOR_FORWARD+90)</fix>

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

<fix/>sensorMotor.duty_cycle_sp = 100</fix>
rightMotor.duty_cycle_sp = 50
leftMotor.duty_cycle_sp = 50
#start calibration mode to sit and recal gyro()

#Create our initial path
currentPath = PathNode(-1, 0)
nodeList.append(currentPath)

forwardAngle = gyroValue() #allow us to monitor the path as we are moving forward
backToLast = False
print "starter: ", forwardAngle
try:
	while not btn.any():
		"""
		Run a thread for detecting left front and right.
		Join thread to be used by main thread which controls direction and motors
		
		"""
		
		#Run sensorThread()
		<fix/>sensorThread();</fix>
		
		#Get values out of sensorThread specifying if theres walls or not
		if(((hasLeft or hasRight) and hasFront) or (hasLeft and hasRight)):
			<fix/>if(backToLast):
				#We're going back to last node
				#The current node is a parent node
				
				#Get the next non-transversed node and check its parent is current
				currentIndex = -1</fix>
				for i, j in enumerate(nodeList):
						<fix/>if j.parentID == current.ID and not j.visited:
							currentIndex = i
				
				if(currentIndex != -1):
					#We found a node on same parent to traverse!
					startPath(nodeList[currentIndex])
					backToLast = False
				else:
					#Our node doesn't have any more directions to go
					#Need to go back again so dont set backToLast
					#Not sure how to get which intersection to take...
					pass
			else:
				#There a new intersection, we need to create a correct amount of nodes
				currentPath.visited = True;
				#current index
				currentIndex = 0
				#work right to left so ensure we turn left to right
				if(hasLeft):
					newPath = PathNode(currentPath.ID, -90)
					print "Creating left node"
					#insert the node after our current parent
					for i, j in enumerate(nodeList):
						if j.ID == newPath.parentID:
							nodeList.insert(i+1,newPath)
							currentIndex = i
				if(hasFront):
					newPath = PathNode(currentPath.ID, 0)
					print "Creating front node"
					#insert the node after our current parent
					for i, j in enumerate(nodeList):
						if j.ID == newPath.parentID:
							nodeList.insert(i+1,newPath)
							currentIndex = i
				if(hasRight):
					newPath = PathNode(currentPath.ID, 90)
					#print currentPath.ID, " ", newPath.parentID, " ", newPath.ID
					print "Creating right node"
					#insert the node after our current parent
					for i, j in enumerate(nodeList):
						if j.ID == newPath.parentID:
							nodeList.insert(i+1,newPath)
							currentIndex = i
				
				#Inside the node we store our travel angle and other data...
				#Our decision is always to take the left-most path
				currentPath = nodeList[currentIndex+1]
				print "New current node:", currentPath.ID, ", parent: ", nodeList[currentIndex].ID
				#for i in range(len(nodeList)):
				#	print "Node ID: ", nodeList[i].ID
				print "Node list length: ", len(nodeList)
				
				#begin traversing the path
				startPath(currentPath)
		elif(hasLeft or hasRight):
			#We need to turn the robot but no need to create a path/intersection
			if(hasRight):
				print "Turning right!"
				turnCorner(90)
			else:
				print "Turning left!"
				turnCorner(-90)
		elif(not hasLeft and not hasFront and not hasRight):
			print "We are at a dead end, turn back!"
			#Need to go back to the last node and check the next value...
			#Go back to parent and do the second/third nodes.
			#If they don't exist (next node isnt sequential ID number) then go
			#back to the again parent node etc... while/for loops
			currentPath.visited = True
			turnCorner(180)
			parentIndex = 0
			#Search for parents index
			for i, j in enumerate(nodeList):
					if j.ID == currentPath.parentID:
						parentIndex = i</fix>
			
			#Set variable to say next time we get to intersection we want to traverse the next node.
			#Set current to parent
			backToLast = True
			currentPath = nodeList[parentIndex]
		else:
			moveFoward(47,47)
			<fix/>detectSkew(gyroValue())
		
		#detectSkew(gyroValue())</fix>
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