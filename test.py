import cv2
import numpy as np
from matplotlib import pyplot as plt
from helper import Helper
from helper import Line
from helper import Point

points = [Point(0, 1), Point(0, 2), Point(0, 3)]

line = Line(points[0], points[2])

print(points[1].onLine(line))