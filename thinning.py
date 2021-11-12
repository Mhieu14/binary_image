import glob
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_WHITE = 255 # fore ground value


def neighbourst(image, i, j):
	shape0 = image.shape[0]
	shape1 = image.shape[1]
	p0 = image[i][j+1]
	p1 = image[i-1][j+1]
	p2 = image[i-1][j]
	p3 = image[i-1][j-1]
	p4 = image[i][j-1]
	p5 = image[i+1][j-1]
	p6 = image[i+1][j]
	p7 = image[i+1][j+1]
	return [p0,p1,p2,p3,p4,p5,p6,p7]

def neighbours(x,y,image):
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

def transitions(neighbours):
	n = neighbours + neighbours[0:1]
	# return sum( (n1, n2) == (1, 0) for n1, n2 in zip(n, n[1:]) )
	count = 0
	for i in range(0, len(n)-1):
		if(n[i] == 1 and n[i+1] == 0): 
			count = count + 1
	return count

def foregroundPixels(image):
	fgp = 0
	row, col = image.shape
	for i in range(2, row-1):
			for j in range(2, col-1):
				if(image[i][j]==1):
					fgp = fgp + 1
	return fgp


def zsAlgoIterationV1(image, dt_value):
	logic_image = np.zeros_like(image)
	logic_image[image == dt_value] = 1
	changing1 = changing2 = 1
	i = 0
	while changing1 or changing2:
		timeStart = time.time()
		changes_occured = 0
		changing1 = []
		rows, columns = logic_image.shape
		for x in range(1, rows - 1):
			for y in range(1, columns - 1):
				# P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, logic_image)
				P0,P1,P2,P3,P4,P5,P6,P7 = n = neighbourst(logic_image, x, y)
				if (logic_image[x][y] == 1
						and 2 <= sum(n) <= 6
						and transitions(n) == 1 
						and P0 * P2 * P6 == 0 
						and P0 * P4 * P6 == 0):
					changing1.append((x,y))
		for x, y in changing1: 
			logic_image[x][y] = 0
			changes_occured = changes_occured + 1
		
		changing2 = []
		for x in range(1, rows - 1):
			for y in range(1, columns - 1):
				# P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, logic_image)
				P0,P1,P2,P3,P4,P5,P6,P7 = n = neighbourst(logic_image, x, y)
				if (logic_image[x][y] == 1 
						and 2 <= sum(n) <= 6 
						and transitions(n) == 1 
						and P0 * P2 * P4 == 0  
						and P2 * P4 * P6 == 0):
					changing2.append((x,y))    
		for x, y in changing2: 
			logic_image[x][y] = 0
			changes_occured = changes_occured + 1
		i = i + 1
		timeEnd = time.time()
		print("Iteration: ", i , "changes_occured: ", changes_occured, "time: ", timeEnd - timeStart)
	return logic_image * dt_value


def zsAlgoIterationV2(image, dt_value):
	logic_image = np.zeros_like(image)
	logic_image[image == dt_value] = 1
	changing1 = changing2 = 1
	i = 0
	rows, columns = logic_image.shape
	setObjectPixels = set()
	for x in range(1, rows - 1):
		for y in range(1, columns - 1):
			if (logic_image[x][y] == 1):
				setObjectPixels.add((x, y))
	while changing1 or changing2:
		timeStart = time.time()
		changes_occured = 0
		changing1 = []
		for x, y in setObjectPixels:
			P0,P1,P2,P3,P4,P5,P6,P7 = n = neighbourst(logic_image, x, y)
			if (2 <= sum(n) <= 6 
					and transitions(n) == 1 
					and P0 * P2 * P6 == 0 
					and P0 * P4 * P6 == 0):
				changing1.append((x,y))
		for x, y in changing1: 
			logic_image[x][y] = 0
			changes_occured = changes_occured + 1
			setObjectPixels.discard((x,y))
		
		changing2 = []
		for x, y in setObjectPixels:
			P0,P1,P2,P3,P4,P5,P6,P7 = n = neighbourst(logic_image, x, y)
			if (2 <= sum(n) <= 6  
					and transitions(n) == 1 
					and P0 * P2 * P4 == 0  
					and P2 * P4 * P6 == 0):
				changing2.append((x,y))    
		for x, y in changing2: 
			logic_image[x][y] = 0
			changes_occured = changes_occured + 1
			setObjectPixels.discard((x,y))
		i = i + 1
		timeEnd = time.time()
		print("Iteration: ", i , "changes_occured: ", changes_occured, "time: ", timeEnd - timeStart)
	return logic_image * dt_value
