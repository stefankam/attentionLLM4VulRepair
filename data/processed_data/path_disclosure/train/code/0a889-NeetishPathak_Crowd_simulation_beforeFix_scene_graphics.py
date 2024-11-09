import time
from graphics import *

<vul/>settingspath = os.getcwd() + '/test2.txt'
# settingspath = os.getcwd() + '/test1.txt'</vul>
framepath = os.getcwd() + '/frame.txt'

agent = []
building = []
triangle = []
road = []
obstacle = []
polygon = []
POI = []

def traverserMatrixDP(matrix):
    x = len(matrix[0])
    rowM = [0 for i in range(0,x+1)]
    myMat = []
    myMat.append(rowM)
    for m in range(0,x):
        myMat.append([0]+matrix[m])
#     print myMat
    n = len(myMat[0])
#     cost = []
#     rowX = [0 for i in range(0,len(matrix[0]))]
#     for i in range(0,x):
#         cost.append(rowX)
#     print cost
    <vul/>cost = [[0]*len(matrix[0]) for i in range(len(matrix[0]))]
    print cost
    print n
    routes = [[0]*len(matrix[0]) for i in range(len(matrix[0]))]</vul>
    
    <vul/>for row in range(len(matrix[0])):
        route = []
        for i in range(row):
            route.append(i)
#         routes[row] = 
    
    for i in range(0,len(matrix[0])):
        for j in range(0,len(matrix[0])):
            if(i==0 or j==0):
                cost[i][j] = 1</vul>
            else:
                <vul/>cost[i][j] = (cost[i-1][j]+cost[i][j-1])</vul>
    print cost
    

def traverSeMatrix(matrix,i,j,m,n,path):
    
    if(i == m-1):
        paux = list(path)
        for k in range(j,n):
#             print "A1: " + str(i*n + k)
            paux.append(i*n + k)
#         print paux
#         for l in range(0,n-j+1):
        for l in range(0,len(paux)):   
            print str(paux[l]) + " ",
        print ""
        return
    
    if(j == n-1):
        paux = list(path)
        for k in range(i,m):
            paux.append(k*n+j)
#         for l in range(0,m-i+1):
        for l in range(0,len(paux)):
            print str(paux[l]) + " ",
        print ""
        return
    
    path.append((i*n)+j)
#     print "I will add at " + str(len(path)-1) + " value: " + str(path[len(path)-1])
    
    traverSeMatrix(matrix, i+1, j, m, n, path)
    
    traverSeMatrix(matrix, i, j+1, m, n, path) 
             
    return

def draw():
    settings = open(settingspath)
    settings.readline()
    while True:
        line = settings.readline()
        if line=="":
            break;
        line = line.split()
        x = float(line[1])
        y = float(line[2])
        if line[0]=="poi":
            num = line[3]
            POI.append([[x,y,num]])
        if line[0]=="agent":
            rad = float(line[3])
            agent.append([[x,y,rad]])
        if line[0]=="obstacle":
            rad = float(line[3])
            type = str(line[4])
            obstacle.append([[x,y,rad, type]])
        if line[0]=="building":
            l = float(line[3])
            b = float(line[4])
            building.append([[x,y,l,b,line[5]]])
        if line[0]=="triangle":
            p2 = float(line[3])
            p3 = float(line[4])
            p4 = float(line[5])
            p5 = float(line[6])
            triangle.append([[x,y,p2,p3,p4,p5,line[6]]])
        if line[0] =="polygon":
            p2 = float(line[3])
            p3 = float(line[4])
            p4 = float(line[5])
            p5 = float(line[6])
            p6 = float(line[7])
            p7 = float(line[8])
            polygon.append([[x,y,p2,p3,p4,p5,p6,p7,line[9]]])
        if line[0]=="road":
            l = float(line[3])
            b = float(line[4])
            h = float(line[5])
            road.append([[x,y,l,b,h]])
    settings.close()
    <vul/>print a</vul>gent
    print obstacle
    print building
    print road
    print triangle
    print polygon
    print POI
    draw_graphics()

    
def draw_graphics():
    print "graphics library"
    positions=[]
#     window=GraphWin("Crowd Simulation",width=1200, height=600)
    <vul/>window=GraphWin("Crowd Simulation",width=400, height=400)</vul>
    window.setBackground('#636363')
    <vul/>window.setCoords(0,0,40,40)</vul>

    for a in agent:
        val = Circle( Point(a[0][0],a[0][1]), a[0][2]-0.1)
        val.setFill("red")
        positions.append([a[0][0],a[0][1]])
        val.draw(window)
        a.append(val)
    
    for t in triangle:
        p1 = Point(t[0][0],t[0][1])
        p2 = Point(t[0][2],t[0][3])
        p3 = Point(t[0][4],t[0][5])
        poly = Polygon(p1,p2,p3)
        poly.setFill("#addd8e")
        poly.draw(window)
        t.append(poly)
    
    for p in polygon:
        p1 = Point(p[0][0],p[0][1])
        p2 = Point(p[0][2],p[0][3])
        p3 = Point(p[0][4],p[0][5])
        p4 = Point(p[0][6],p[0][7])
        cent_x = (p[0][0] + p[0][2] + p[0][4] + p[0][6])/4.0
        cent_y = (p[0][1] + p[0][3] + p[0][5] + p[0][7])/4.0
        poly = Polygon(p1,p2,p3,p4)
        poly.setFill("#bf78ce")
        poly.draw(window)
        if p[0][8] == "River":
            poly.setFill("#99d8c9")
        text = Text(Point(cent_x,cent_y),p[0][8])
        text.setSize(8)
        text.draw(window)
        p.append(poly)
    
    for a in building:
        bd1 = Point(a[0][0], a[0][1])
#         bd2 = Point(a[0][0], a[0][1] + a[0][2])
        bd3 = Point(a[0][0] + a[0][3], a[0][1] + a[0][2])
#         bd4 = Point(a[0][0] + a[0][3], a[0][1])

        #Line(bd1,bd2).draw(window)
        #Line(bd2,bd3).draw(window)
        #Line(bd3,bd4).draw(window)
        #Line(bd4,bd1).draw(window)
        
        rect = Rectangle(bd1,bd3)
        if "College" == str(a[0][4]):
            color = 'Brown'
        elif 'Hospital' == str(a[0][4]):
            color = 'Blue'
        elif "Forest" == str(a[0][4]):
            color = "Green"
        elif "Pub" == str(a[0][4]):
            color = "Yellow"
        elif "Home" == str(a[0][4]):
            color = "White"
        elif "Office" == str(a[0][4]):
            color = "Magenta"
        elif "Gym" == str(a[0][4]):
            color = "Maroon"
        elif "Pool" == str(a[0][4]):
            color = "cyan"
        elif "Park" == str(a[0][4]):
            color = "Green"
        elif "Hostel" == str(a[0][4]):
            color = "Yellow"
        elif "Bridge" == str(a[0][4]):
            color = "Brown"
        elif "Mall" == str(a[0][4]):
            color = "#5ab4ac"
        elif "Restaurant" == str(a[0][4]):
            color = "#d8b365"
        else:
            color = "#addd8e"
        rect.setFill(color)
        rect.draw(window)
        anchorPoint = rect.getCenter()
        text = Text(anchorPoint, a[0][4])
        text.setSize(8)
        text.draw(window)
        a.append(rect)
#     time.sleep(10)

    for r in road:
        r1= Point(r[0][0],r[0][1])
        r2 = Point(r[0][0] + r[0][3],r[0][1]+r[0][2])
        rd = Rectangle(r1,r2)
        rd.setFill("#636363")
        rd.setOutline("#636363")
        rd.draw(window)
        r.append(rd)

    for o in obstacle:
        val = Circle( Point(o[0][0],o[0][1]), o[0][2])
        type = o[0][3]
        if type == "road":
            val.setFill("#636363")
            val.setOutline("#636363")
        else:
            val.setFill("black")
        val.draw(window)
        o.append(val)
    for poi in POI:
        poi1 = Point(poi[0][0], poi[0][1])
        <vul/>poi2 = Point(poi[0][0] + 5, poi[0][1] + 5)</vul>
        rct_poi = Rectangle(poi1,poi2)
        rct_poi.setFill("Red")
        rct_poi.draw(window)
        text = Text(rct_poi.getCenter(),poi[0][2])
        text.setSize(8)
        text.draw(window)
    matrix = []
    for x in range(0,40,5):
        m1 = []
        for y in range(0,40,5):
            m1.append((x*8 + y)/5)
            rP = Rectangle(Point(x,y),Point(x+5,y+5))
            rP.setOutline("black")
            rP.draw(window)
            t = Text(rP.getCenter(),(x*8 + y)/5)
            t.setSize(8)
            t.draw(window)
        matrix.append(m1)
    
    for i in range(0,8):
        for j in range(0,8):
            print str(matrix[i][j])  + " ",
        print "\n"
    <vul/>dim = 4</vul>
    a = []
    for l in range(0,dim):
        b = []
        for m in range(0,dim):
            <vul/>b.append(l*dim+m)</vul>
        a.append(b)
        
    print a
    path = []
    <vul/>traverSeMatrix(a,0,0,dim,dim,path)</vul>       
    traverserMatrixDP(a)
    
            
    try: 
        file = open(framepath)
    except TypeError:
        print "cannot find frame file"
    file.readline()
    while True:
        line = file.readline()
        if line=="":
            break;
        line = line.split()
        i= int(line[0])
        newpos = [float(line[1]),float(line[2])]
        dpos = [newpos[0]-positions[i][0],newpos[1]-positions[i][1]]
        agent[i][1].move(dpos[0],dpos[1])
        positions[i]=newpos
        time.sleep(0.1)
    file.close()
    window.getMouse()
draw()
