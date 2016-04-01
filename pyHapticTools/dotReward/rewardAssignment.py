#!/usr/local/bin/python
'''
Program runs to longitudinal locaiton of edges('zipper')
in biotac from http server request with image tag in http
body

'''
# opencv dot detections
from collections import deque
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import math
import argparse
import imutils
import cv2
import time



def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)

	# return the image with the contour number drawn on it
	return image

def findOffsetForRewardAssignment(filename):
	start_time = time.time()
	redLower  = (0, 89, 91)
	redUpper  = (10, 241, 241)
	blueLower = (98, 28, 50)
	blueUpper = (129, 191, 198)
	
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image",
	# 	help="path to the (optional) video file")
	# args = vars(ap.parse_args())
	# # load image and do standard guassianBlur and transform to HSV
	# if args['image']:
	#     image = cv2.imread(args['image'])
	#     image = imutils.resize(image, width=1500)
	#     blurred = cv2.GaussianBlur(image, (11, 11), 0)
	#     frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# else:
    #	  camera = cv2.VideoCapture(0)

	image = cv2.imread(filename)
	image = imutils.resize(image, width=1500)
	image=cv2.flip(image,-1)
	blurred = cv2.GaussianBlur(image, (11, 11), 0)
	frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


	
	imageOrig = image.copy()
	print "Image size "
	print image.shape
	# Start Biotac point find
	mask = cv2.inRange(frame, redLower, redUpper)
	#mask = cv2.erode(mask, None, iterations=1)
	mask = cv2.dilate(mask, None, iterations=1)
	
	#k-means 
	x, y = np.where(mask > 2)
	X = np.zeros((len(x),2))
	X[:,1] = x
	X[:,0] = y
	
	est = KMeans(n_clusters=2).fit(X)
	labels = est.labels_

	center =  est.cluster_centers_ 
	for c in center:
		cen = [ int(round(elem, 0)) for elem in c ]
		print cen
		cv2.circle(image, tuple(cen), 2, (0, 255, 0), -1)
	output = cv2.bitwise_and(image, image, mask = mask)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]

	# find blue bag
	bagMask = cv2.inRange(frame, blueLower, blueUpper)
	bagOutput = cv2.bitwise_and(image, image, mask = bagMask)

	dotsCenter =  [int(round(np.mean(center[:,0]))), int(round(np.mean(center[:,1]))) ]
	print 'The center of the dots:', (dotsCenter)

	maskDot = np.zeros(image.shape[:2], dtype="uint8")
	maskDot[dotsCenter[1]-100:dotsCenter[1]+100, dotsCenter[0]-150:dotsCenter[0]+150] = 255
	bagImage = cv2.bitwise_and(bagMask, bagMask, mask=maskDot)
	bagImageDisplay = cv2.bitwise_and(image, image, mask=bagImage)
	BagCnts = cv2.findContours(bagImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	
	for c in BagCnts[:2]:
		rect = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(bagImageDisplay,[box],0,(255,0,255),2)
	
	BagCnts = np.vstack(BagCnts)

	rect = cv2.minAreaRect(BagCnts)
	box = cv2.cv.BoxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(bagImageDisplay,[box],0,(255,255,255),1)

	rows,cols = image.shape[:2]
	[vx,vy,x,y] = cv2.fitLine(BagCnts, cv2.cv.CV_DIST_L2,0,0.01,0.01)
	print vx, vy, x, y
	lefty = int((-x*vy/vx) + y)
	righty = int(((cols-x)*vy/vx)+y)
	print 'lefty'
	print lefty
	print 'righty'
	print righty
	cv2.line(image,          (cols-1,righty),(0,lefty),(0,215,255),2)
	cv2.line(bagImageDisplay,(cols-1,righty),(0,lefty),(0,215,255),2)

	# find the horizontal location of the bag in the biotac
	# Use equation of a line to find at dot center
	m = float(righty - lefty) / float( cols-1 ) 
	b = lefty 
	print 'dotsCenter[0] - ',dotsCenter[0]
	y = m* float(dotsCenter[0]) + float(b)
	cv2.circle(image, (dotsCenter[0], int(round(y))), 2, (255, 0, 0), -1)

	
	pixelsBetweenDots = int(round(math.sqrt( (center[0][0] - center[1][0])**2 + (center[0][1] - center[1][1])**2 )))
	# measued separation between two red dots mounted on Biotac
	DOT_SEPARATION_DISTANCE = 1.2
	cmPPixel = DOT_SEPARATION_DISTANCE / pixelsBetweenDots

	bagOffset = float(dotsCenter[1] - y) *  cmPPixel
	print 'bagOffset(mm) - ', bagOffset*10
	textXshift = 800
	cv2.putText(image, "bagOffset: %6.4fmm" % (bagOffset * 10), (image.shape[1] - textXshift,
		image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0), 2)
	cv2.startWindowThread() 

	#cv2.imshow("Original", image)
	cv2.imshow("BagOffsetRewardAssignment", np.hstack([imageOrig, image]))
	#cv2.imshow("BioTac", output )
	#cv2.imshow("Bag", bagMask )
	#cv2.imshow("BagMask", bagImageDisplay )
	posIndex = filename[::-1].find('/')
	filename = filename[:-posIndex] + 'rewardJpg/' + filename[-posIndex:];
	filename = filename[:-4] + '_' + 'mm%4.2f' % (bagOffset * 10)
	print '###fileName:' + filename
	cv2.imwrite(filename + '.jpg',np.hstack([imageOrig, image]))
	print("--- %s seconds ---" % (time.time() - start_time))
	#cv2.moveWindow("test", 1300, -1000 )
	cv2.waitKey(1)


