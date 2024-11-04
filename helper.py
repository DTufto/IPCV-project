import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from enum import Enum

kernel = np.ones((3, 3), np.uint8) 

EMPTYPOINT = "EMPTYPOINT"
LCORNER = "LCORNER"
LGOALLINEX16M = "LGOALLINEX16M"
LGOALLINEX5M = "LGOALLINEX5M"
LM16 = "LM16"
LM5 = "LM5"
LGOALPOST = "LGOALPOST"
RGOALPOST = "RGOALPOST"
RM5 = "RM5"
RM16 = "RM16"
RGOALLINEX5M = "RGOALLINEX5M"
RGOALLINEX16M = "RGOALLINEX16M"
RCORNER = "RCORNER"

EMPTYLINE = "EMPTY"
GOALLINE = "GOALLINE"
SIDELINEL = "LEFTSIDELINE"
SIDELINER = "RIGHTSIDELINE"
LINE16L = "LINE16L"
LINE16R = "LINE16R"
LINE16 = "LINE16"
LINE5L = "LINE5L"
LINE5R = "LINE5R"
LINE5 = "LINE5"

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def angle(self):
        if self.x == 0:
            return 90.0
        return math.degrees(math.atan(self.y/self.x))
    
    def mag(self):
        power = (self.x**2) + (self.y**2) + (self.z**2)
        # print(type(self.x))
        # print(str(self.x) + " : " + str(self.y) + " : " + str(self.z) + " : " + str(power))
        return math.sqrt(power)
    
    def __str__(self):
        return f"Vector with x: {self.x}, y: {self.y} and z: {self.z}"
    
    def __mul__(self, o): # Dot product
        return (self.x * o.x + self.y * o.y + self.z * o.z)
    
    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)
    
    def __le__(self, o):
        # print(self.mag())
        return self.mag() <= o
    
    def __and__(self, o):# Cross product
        return Vector(self.y * o.z - self.z * o.y, -1*(self.x * o.z - self.z * o.x), self.x * o.y - self.y * o.x)

class Point:
    def __init__(self, x, y, z=0):
        self.x = np.int64(x)
        self.y = np.int64(y)
        self.z = np.int64(z)
        self.interLines = []
        self.name = EMPTYPOINT
    
    def __add__(self, o):
        return Vector(self.x + o.x, self.y + o.y, self.z + o.z)
    
    def __sub__(self, o):
        return Vector(self.x - o.x, self.y - o.y, self.z - o.z)
    
    def __str__(self):
        return f"Point named: {self.name} with x: {self.x} and y: {self.y} and z: {self.z}"
    
    def onLine(self, line, epsilon=10000):
        vec0 = line.points[0] - line.points[1]
        vec1 = line.points[1] - self
        crossResult = (vec0 & vec1)
        # print(crossResult.mag())
        if crossResult <= epsilon:
            return True
        return False
    
    def withinLineSquare(self, line, bounds=10):
        maxY = max(line.points[0].y, line.points[1].y)
        minY = min(line.points[0].y, line.points[1].y)
        if self.y >= minY - bounds and self.y <= maxY + bounds:
            maxX = max(line.points[0].x, line.points[1].x)
            minX = min(line.points[0].x, line.points[1].x)
            if self.x >= minX - bounds and self.x <= maxX + bounds:
                return True
        return False
    
    def betweenLinePoints(self, line):
        return self.onLine(line) and self.withinLineSquare(line)
    
class Line:
    def __init__(self, line):
        x1, y1, x2, y2 = line[0]
        if y1 > y2:
            x2, y2, x1, y1 = line[0]
        self.points = [Point(x1, y1), Point(x2, y2)]
        self.angle = (self.points[1] - self.points[0]).angle()
        self.length = (self.points[1] - self.points[0]).mag()
        self.name = EMPTYLINE
        self.intersections = []

    def __lt__(self, o):
        return self.angle < o.angle
        
    def __neg__(self):
        return self.points[0].x, self.points[0].y
    
    def __pos__(self):
        return self.points[1].x, self.points[1].y

    def __str__(self):
        return f"Line named: {self.name} \nFrom: \n{self.points[0]} \nTo: \n{self.points[1]} \nWith angle: {self.angle}"

    def decompose(self):
        p0 = self.points[0]
        p1 = self.points[1]
        a = (p0.y-p1.y)/(p0.x-p1.x)
        b = (p0.x*p1.y - p1.x*p0.y)/(p0.x-p1.x)
        return (a, b)

    def __mul__(self, o):
        l0a, l0b = self.decompose()
        l1a, l1b = o.decompose()
        # Find intersection
        A = np.array([[-l0a, 1], [-l1a, 1]])
        b = np.array([[l0b], [l1b]])
        # you have to solve linear System AX = b where X = [x y]'
        X = np.linalg.pinv(A) @ b
        x, y = np.round(np.squeeze(X), 4)
        return Point(x, y)
    
    def extendsLine(self, other):
        point0 = other.points[0]
        point1 = other.points[1]
        delOther = False
        if point0.onLine(self) and point1.onLine(self):
            if point0.y < self.points[0].y:
                self.points[0] = point0
            if point1.y > self.points[1].y:
                self.points[1] = point1
            delOther = True
        return delOther

class Helper:
    def __init__(self):
        return

    def dilate(self, img, iters=1):
        return cv2.dilate(img, kernel, iterations=iters)

    def erode(self, img, iters=1):
        return cv2.erode(img, kernel, iterations=iters)

    def houghLines(self, img):
        img = cv2.GaussianBlur(img,(5,5),0)
        ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.CV_8U)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the edges in the image using canny detector
        # edges = cv2.Canny(gray, 120, 240)
        # Detect points that form a line
        lines = cv2.HoughLinesP(gray, 3, np.pi/720, 1000, minLineLength=200, maxLineGap=40)
        return lines
    
    def houghPalen(self, img):
        img = cv2.GaussianBlur(img,(5,5),0)
        ret,img = cv2.threshold(img,210,255,cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.CV_8U)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the edges in the image using canny detector
        # edges = cv2.Canny(gray, 120, 240)
        # cv2.imshow("blank", gray)
        # cv2.waitKey(50000)
        # Detect points that form a line
        lines = cv2.HoughLinesP(gray, 3, np.pi/3600, 200, minLineLength=40, maxLineGap=10)
        return lines
    
    def satAdjHSV(self, imgBGR, sAdj):
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(imgHSV.astype('int'))
        s = s*sAdj
        s = np.clip(s,0,255)
        return cv2.merge([h,s,v]).astype('uint8')
    
    def thresholdBand(self, img, lower, upper):
        _, mask0 = cv2.threshold(img,lower,255,cv2.THRESH_BINARY)
        _, mask1 = cv2.threshold(img,upper,255,cv2.THRESH_BINARY_INV)
        return cv2.bitwise_and(mask0, mask0, mask=mask1)
    
    def fieldMask(self, imgHSV):
        (h, s, v) = cv2.split(imgHSV)       
        mask = self.thresholdBand(h, 40, 70)
        mask = self.erode(mask, 14)
        mask = self.dilate(mask, 50)
        return mask

    def lineGrouper(self, aList):
        aList.sort()
        retList = []
        buildList = []
        groupLow = aList[0].angle
        for elem in aList:
            if abs(elem.angle - groupLow) < 10:
                buildList.append(elem)
            else:
                groupLow = elem.angle
                retList.append(buildList.copy())
                buildList.clear()
                buildList.append(elem)
        if (len(buildList) > 0):
            retList.append(buildList.copy())
        return retList
    
    def lineJoiner(self, aGroupedList):
        toDelete = []
        for group in aGroupedList:
            indexToRemove = []
            for index, line in enumerate(group):
                i = index + 1
                while True:
                    if i < len(group):
                        if line.extendsLine(group[i]):
                            group.remove(group[i])
                        else:
                            i += 1
                    else:
                        break
        return aGroupedList

    def lineFilter(self, lines, maxlength, minlength, minAngle, maxAngle):
        retList = []
        for line in lines:
            if line.length > minlength and line.length < maxlength:
                if line.angle > minAngle and line.angle < maxAngle:
                    retList.append(line)
        return retList

    def createPointDict(self, lines):
        points = []
        for vertical in lines[0]:
            for horizontal in lines[1]:
                intersect = vertical * horizontal
                if intersect.betweenLinePoints(vertical) and intersect.betweenLinePoints(horizontal):
                    vertical.intersections.append(intersect)
                    horizontal.intersections.append(intersect)
                    intersect.interLines.append(vertical)
                    intersect.interLines.append(horizontal)
                    points.append(intersect)
        for line in lines[0]:
            line.intersections.sort(key = lambda x : (line.points[0] - x).mag())
        for line in lines[1]:
            line.intersections.sort(key = lambda x : (line.points[0] - x).mag())
        return points
        
    def classifyLines(self, lines):
        goalGroup = -1
        horiGroup = -1
        goalLine = None
        for line in lines[0]:
            if len(line.intersections) >= 3:
                line.name = GOALLINE
                goalGroup = 0
                horiGroup = 1
                goalLine = line
        for line in lines[1]:
            if len(line.intersections) >= 3:
                line.name = GOALLINE
                goalGroup = 1
                horiGroup = 0
                goalLine = line
        lCorner = None
        rCorner = None
        for line in lines[horiGroup]:
            if len(line.intersections) == 1:
                if line.angle - goalLine.angle < 180:
                    line.name = SIDELINEL
                    line.intersections[0].name = LCORNER
                    lCorner = line.intersections[0]
                else:
                    line.name = SIDELINER
                    line.intersections[0].name = RCORNER
                    rCorner = line.intersections[0]
        LtoR = [LGOALLINEX16M, LGOALLINEX5M, RGOALLINEX5M, RGOALLINEX16M]
        RtoL = [RGOALLINEX16M, RGOALLINEX5M, LGOALLINEX5M, LGOALLINEX16M]
        line16 = [LINE16L, LINE16, LINE16R]
        line5 = [LINE5L, LINE5, LINE5R]
        if goalLine.intersections[0].name == LCORNER:
            for index, inter in enumerate(goalLine.intersections[1:]):
                inter.name = LtoR[index]
                if inter.name == LGOALLINEX16M:
                    for line in inter.interLines:
                        if line.name == EMPTYLINE:
                            line.name = LINE16L
                            for inter in line.intersections:
                                if inter.name == EMPTYPOINT:
                                    inter.name = LM16
                                    for line in inter.interLines:
                                        if line.name == EMPTYLINE:
                                            line.name = LINE16
                                            for inter in line.intersections:
                                                if inter.name == EMPTYLINE:
                                                    inter.name = RM16
                                                    for line in inter.interLines:
                                                        if line.name == EMPTYLINE:
                                                            line.name = LINE16R
                elif inter.name == LGOALLINEX5M:
                    for line in inter.interLines:
                        if line.name == EMPTYLINE:
                            line.name = LINE5L
                            for inter in line.intersections:
                                if inter.name == EMPTYPOINT:
                                    inter.name = LM5
                                    for line in inter.interLines:
                                        if line.name == EMPTYLINE:
                                            line.name = LINE5
                                            for inter in line.intersections:
                                                if inter.name == EMPTYLINE:
                                                    inter.name = RM5
                                                    for line in inter.interLines:
                                                        if line.name == EMPTYLINE:
                                                            line.name = LINE5R

        elif goalLine.intersections[0].name == RCORNER:
            for index, inter in enumerate(goalLine.intersections[1:]):
                inter.name = RtoL[index]
                if inter.name == RGOALLINEX16M:
                    for line in inter.interLines:
                        if line.name == EMPTYLINE:
                            line.name = LINE16R
                            for inter in line.intersections:
                                if inter.name == EMPTYPOINT:
                                    inter.name = RM16
                                    for line in inter.interLines:
                                        if line.name == EMPTYLINE:
                                            line.name = LINE16
                                            for inter in line.intersections:
                                                if inter.name == EMPTYLINE:
                                                    inter.name = LM16
                                                    for line in inter.interLines:
                                                        if line.name == EMPTYLINE:
                                                            line.name = LINE16L
                elif inter.name == RGOALLINEX5M:
                    for line in inter.interLines:
                        if line.name == EMPTYLINE:
                            line.name = LINE5R
                            for inter in line.intersections:
                                if inter.name == EMPTYPOINT:
                                    inter.name = RM5
                                    for line in inter.interLines:
                                        if line.name == EMPTYLINE:
                                            line.name = LINE5
                                            for inter in line.intersections:
                                                if inter.name == EMPTYLINE:
                                                    inter.name = LM5
                                                    for line in inter.interLines:
                                                        if line.name == EMPTYLINE:
                                                            line.name = LINE5L