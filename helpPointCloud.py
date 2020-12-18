'''
File containing auxiliary functions to help with
creating the point cloud outside of the classes.

Some of the code comes directly from class material.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from time import time
import plotly.graph_objects as go

def detectFeatures(img1, img2, featureDetection):
	'''
	This function uses the code from the tutorial to
	detect feature in each of the images.
	'''
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
	'''
	This function uses the code from the tutorial to find a list of matching features
	between two images and returns them as two corresponding lists.
	'''
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
	return src_pts, dst_pts

def drawKeypoints(img1, kp1, img2, kp2):
	'''
	This function uses the code from the tutorial to find a list of matching features
	between two images and returns them as two corresponding lists.
	'''
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
	'''
	This function uses the code from the tutorial to plot matching features
	between two images.
	'''
	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches,
	                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
	                          matchColor=(0, 255, 0))
	plt.imshow(img3), plt.show()

def estimateRelativeExtrinsics(img1, img2, id1, id2, K):
	'''
	This function estimates the relative extrinsics between two images.
	'''

	# This function estimates the relative pose of the camera in img1 wrt img2
	pts1, pts2 = matchFeatures(img1, img2, 'SIFT') # match time 35.29676103591919

	# Find the fundamental matrix F (options: 7pt, 8pt, LMEDS, RANSAC)
	F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
	# This function find fundamental matrix using the normalized 8 point method
	# Solves the homogeneous system involving the point correspondences for F
	# In Defence of the 8-point Algorithm Hartley

	# Find the essential matrix E
	E = K.T @ F @ K # http://robotics.stanford.edu/~birch/projective/node20.html
	retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
	# This function uses SVD to reduce the essential matrix E into
	# a rotation matrix and translation vector.
	# Hartley and Zisserman Multiple View Geometry in Computer Vision
	return R,t, pts1, pts2

def plot_pointCloud(pc):
	'''
	This function plots the point cloud.
	'''
	fig = go.Figure(data=[go.Scatter3d(
		x=pc[:, 0],
		y=pc[:, 1],
		z=-pc[:, 2],
		mode='markers',
		marker=dict(
			size=2,
			color=pc[:, 3:][..., ::-1],
			opacity=0.8
		)
	)])
	fig.show()

def compute3DPoints(listOfMatches, viewSet, isBest):
	'''
	This is a helper function to find the 3D Points.
	'''
	if isBest:
		for m in listOfMatches:
			m.findWorldPoint(viewSet)
	else:
		for m in listOfMatches:
			m.findCameraSpacePoints(viewSet)


def filterRelativeProjections(listOfMatches):
	'''
	This is a helper function to reject outliers.
	'''
	# Get a list of all the projections' norms
	norms = []
	for m in listOfMatches:
		for p in m.getCameraSpacePoints():
			norms.append(np.linalg.norm(p))
	# Reject the outliers that are more than 1.5 * IQR
	# below the first quartile or above the third quartile
	# Find the bounds
	q75, q25 = np.percentile(norms, [75,25])
	iqr = q75 - q25
	lowerBound = q25 - 1.5 * iqr
	upperBound = q75 + 1.5 * iqr
	# John Tukey 1977 Exploratory Data Analysis
	for m in listOfMatches:
		m.filterCameraSpacePoints(lowerBound, upperBound)

def filterWorldPoints(listOfMatches):
	'''
	This is a helper function to remove outliers
	'''
	listOfMatches = [m for m in listOfMatches if m.worldPoint is not None]
	return listOfMatches

def createPointCloud(listOfMatches):
	'''
	This is a helper function to create a point cloud
	'''
	pointCloud = np.zeros((len(listOfMatches), 6))
	for m in range(len(listOfMatches)):
		pointCloud[m][0:3] = listOfMatches[m].worldPoint.reshape(3)
		pointCloud[m][3:] = np.array([255., 0., 0.])
	return pointCloud



