import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

kernel = np.ones((3, 3), np.uint8) 

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def angle(self):
        if self.x == 0:
            return 90.0
        return math.degrees(math.atan(self.y/self.x))
    
    def __str__(self):
        return f"Vector with x: {self.x} and y: {self.y}"

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, o):
        return Vector(self.x + o.x, self.y + o.y)
    
    def __sub__(self, o):
        return Vector(self.x - o.x, self.y - o.y)
    
    def __str__(self):
        return f"Point with x: {self.x} and y: {self.y}"
    
class Line:
    def __init__(self, line):
        x1, y1, x2, y2 = line[0]
        if y1 > y2:
            x2, y2, x1, y1 = line[0]
        self.points = [Point(x1, y1), Point(x2, y2)]
        self.angle = (self.points[0] - self.points[1]).angle()
    
    def __lt__(self, o):
        return self.angle < o.angle
        
    def __neg__(self):
        return self.points[0].x, self.points[0].y
    
    def __pos__(self):
        return self.points[1].x, self.points[1].y

    def __str__(self):
        return f"Line from: \n{self.points[0]} \nTo: \n{self.points[1]} \nWith angle: {self.angle}"

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
        edges = cv2.Canny(gray, 120, 240)
        # Detect points that form a line
        lines = cv2.HoughLinesP(gray, 3, np.pi/720, 1000, minLineLength=200, maxLineGap=40)
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

    def get_line_intersection(self, line1, line2):
        """
        Find intersection point of two lines using their endpoints.
        Returns None if lines are parallel or don't intersect.
        """
        x1, y1 = -line1
        x2, y2 = +line1
        x3, y3 = -line2
        x4, y4 = +line2

        # Calculate denominator
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-8:  # Lines are parallel
            return None

        # Calculate intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        # Check if intersection occurs within line segments
        if not (0 <= t <= 1):
            return None

        # Calculate intersection point
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (int(x), int(y))

    def find_field_intersections(self, linesGrouped, mask, min_distance=20):
        """
        Find all valid intersection points between line groups within the field mask.
        """
        intersections = []

        # Compare each group with every other group
        for i in range(len(linesGrouped)):
            for j in range(i + 1, len(linesGrouped)):
                # Compare each line in first group with each line in second group
                for line1 in linesGrouped[i]:
                    for line2 in linesGrouped[j]:
                        intersection = self.get_line_intersection(line1, line2)

                        if intersection is None:
                            continue

                        x, y = intersection

                        # Check if point is within image bounds and field mask
                        if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and
                                mask[y, x] > 0):

                            # Check if this point is far enough from existing points
                            is_unique = True
                            for existing_x, existing_y in intersections:
                                dist = np.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2)
                                if dist < min_distance:
                                    is_unique = False
                                    break

                            if is_unique:
                                intersections.append((x, y))

        return intersections

    def draw_intersections(self, img, intersections, radius=5, color=(0, 255, 0), thickness=-1):
        """
        Draw the intersection points on the image
        """
        result = img.copy()
        for x, y in intersections:
            cv2.circle(result, (x, y), radius, color, thickness)
        return result
