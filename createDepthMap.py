import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from time import time

def findCameraIntrinsic(idx1, idx2):
	# This function finds the intrinsic matrices for all the images
	return matrixList[idx1], matrixList[idx2]

def detectFeatures(img1, img2, featureDetection):
	if featureDetection == "SIFT":
		detector = cv2.xfeatures2d.SIFT_create()
	elif featureDetection == "SURF":
		detector = cv2.xfeatures2d.SIFT_create()
	elif featureDetection == "ORB":
		detector = cv2.ORB_create(nfeatures=1000)
	
	kp1, desc1 = detector.detectAndCompute(img1, None)
	kp2, desc2 = detector.detectAndCompute(img2, None)
	return kp1, desc1, kp2, desc2

def matchFeatures(img1, img2, featureDetection="SIFT"): # sample code from tutorial
	# This function uses the code from the tutorial to find a list of matching features
	# between two images and returns them as two corresponding lists.
	start = time()
	if featureDetection == "SIFT":
		kp1, desc1, kp2, desc2 = detectFeatures(img1, img2, featureDetection)
	elif featureDetection == "SURF":
		kp1, desc1, kp2, desc2 = detectFeatures(img1, img2, featureDetection)
	elif featureDetection == "ORB":
		kp1, desc1, kp2, desc2 = detectFeatures(img1, img2, featureDetection)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1, desc2, k=2)  # k=2 means find the top two matchs for each query descriptor

	# Apply ratio test (as per David Lowe's SIFT paper: compare the best match with the 2nd best match_
	good_matches = []
	good_matches_without_list = []
	for m, n in matches:
		if m.distance < 0.75 * n.distance:  # only accept matches that are considerably better than the 2nd best match
			good_matches.append([m])
			good_matches_without_list.append(m)
	src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches_without_list]) #.reshape(-1, 2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches_without_list]) #.reshape(-1, 1, 2)
	# plotFeatures(img1, kp1, img2, kp2, good_matches)
	finish = time()
	print(finish - start)
	return src_pts, dst_pts

def drawKeypoints(img1, kp1, img2, kp2):
	img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
	                              color=(255, 255, 0))
	img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
	                              color=(255, 255, 0))
	plt.subplot(1, 2, 1)
	plt.imshow(img1_kp)
	plt.subplot(1, 2, 2)
	plt.imshow(img2_kp)
	plt.show()

def plotFeatures(img1, kp1, img2, kp2, good_matches):
	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches,
	                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
	                          matchColor=(0, 255, 0))
	plt.imshow(img3), plt.show()

def estimateRelativePose(img1, img2, idx1, idx2):
	# This function estimates the relative pose of the camera in img1 wrt img2
	# pts1, pts2 = matchFeatures(img1, img2) # match time 35.29676103591919
	# pts1, pts2 = np.load(outputDir + 'pts1.npy'), np.load(outputDir + 'pts2.npy')
	pts1, pts2 = np.load(outputDir + 'pts1.npy'), np.load(outputDir + 'pts2.npy')
	# Find the fundamental matrix F (options: 7pt, 8pt, LMEDS, RANSAC)
	F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

	'''
	# Code to test the fundamental matrix 
	p1, p2 = np.ones(3), np.ones(3)
	maxErr = 0
	randIndex = np.random.randint(0,len(pts1))
	# print(pts1[1], pts2[1])
	p1[0:2] = pts1[randIndex][0:2]
	p2[0:2] = pts2[randIndex][0:2]
	print(p1 @ F @ p2.reshape(3,1))
	# for i in range(len(pts1)):
	# 	p1[0:2] = pts1[i][0:2]
	# 	p2[0:2] = pts2[i][0:2]
	# 	maxErr = max(maxErr, abs(p1 @ F @ p2.reshape(3,1)))
	# print(maxErr)
	# return
	'''

	# Obtain the camera intrinsic matrix K
	# Normal Scenario
	# 1. Run through the camera calibration pipeline
	# 2. Call findCameraIntrinsic to retrieve camera matrix
	K1, K2 = findCameraIntrinsic(idx1, idx2)
	# print(K1.shape, K2.shape, F.shape)
	# Find the essential matrix E
	E = K2.T @ F @ K1
	# Since K1 and K2 are the same
	retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K1)
	# print(R)
	# print(t)
	return R,t, pts1, pts2

if __name__ == "__main__":
	from os import listdir
	from os.path import isfile, join
	from ViewSet import *

	# set the images directory for semper dataset
	imgDir = "images/semper/"
	outputDir = "output/temp/"

	# get list of file names
	files = sorted([f for f in listdir(imgDir) if isfile(join(imgDir, f))])
	# get list of camera matrix file names
	cameraFiles = [imgDir + f for f in files if "camera" in f]
	# get list of image file names
	imgFiles = [imgDir + f for f in files if "ppm" in f and "camera" not in f and "3D" not in f]
	# convert ppm to numpy array
	images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in imgFiles]
	# load the camera matrices
	matrixFiles = [open(f) for f in cameraFiles]
	matrixList = [np.zeros((3,3)) for f in cameraFiles]
	for i in range(len(matrixFiles)):
		for j in range(3):
			row = matrixFiles[i].readline().split()
			for k in range(3):
				matrixList[i][j,k] = np.float(row[k])
	idx1, idx2 = 0, 1

	# get the relative pose between the two views
	R, t, srcPts, dstPts = estimateRelativePose(images[idx1], images[idx2], idx1, idx2) # estimate time 35.433154821395874

	# instantiate viewSet object
	v = ViewSet()

	# add the first view
	v.addView(0, None, None)

	# add the second view
	v.addView(1, R, t)

	# v.debugViews()

	# add connection
	v.addConnection(0, 1, R, t, srcPts, dstPts)
	# v.debugConnections()
	# find the matches
	listOfMatches = []
	discovered = [dict(),dict()]
	v.findPointTracks(listOfMatches, discovered)
	# find the projections


