import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from time import time

def findCameraIntrinsic(idx1, idx2):
	# This function finds the intrinsic matrices for all the images
	return matrixList[idx1], matrixList[idx2]

def detectSIFTFeatures(img1, img2):
	sift = cv2.xfeatures2d.SIFT_create()
	kp1, desc1 = sift.detectAndCompute(img1, None)
	kp2, desc2 = sift.detectAndCompute(img2, None)
	return kp1, desc1, kp2, desc2

def matchFeatures(img1, img2, featureDetection="SIFT"): # sample code from tutorial
	# This function uses the code from the tutorial to find a list of matching features
	# between two images and returns them as two corresponding lists.
	if featureDetection == "SIFT":
		kp1, desc1, kp2, desc2 = detectSIFTFeatures(img1, img2)

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

def estimateRelativePose(img1, img2, idx1, idx2):
	# This function estimates the relative pose of the camera in img1 wrt img2
	# pts1, pts2 = matchFeatures(img1, img2) # match time 35.29676103591919
	pts1, pts2 = np.load(outputDir + 'pts1.npy'), np.load(outputDir + 'pts2.npy')
	# Find the fundamental matrix F (options: 7pt, 8pt, LMEDS, RANSAC)
	F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
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
	return R,t

	# # Get the singular value decomposition to find
	# # Rotation matrix R and translation vector t
	# U, Sigma, V = np.linalg.svd(E)
	# Sigma = np.diag(Sigma)
	#
	# # Mathematical postprocessing
	# # Sigma is supposed to be of the form
	# # s 0 0 We can make the last entry zero
	# # 0 s 0 and take the average of values of s
	# # 0 0 0
	# # s = np.average(Sigma[0,0], Sigma[1,1])
	# # Sigma[0,0] = s; Sigma[1,1] = s
	# print(U)
	# print(Sigma)
	# print(V)
	#
	# W = np.array([[0, -1, 0],
	#               [1,  0, 0],
	#               [0,  0, 1]])
	#
	# Winv = W.T
	# Z = np.array([[ 0, 1, 0],
	#               [-1, 0, 0],
	#               [ 0, 0, 0]])
	#
	# # Solve for the rotation matrix and translation vectors
	# # Note there are four solutions
	# T_cross = U @ W @ Sigma @ U.T
	#
	# # Since Sigma may not work on real world data
	# T_cross_alt = U @ Z @ U.T
	#
	# # There are 2 possible rotation matrices
	# R1 = U @ Winv @ V.T
	# R2 = U @ W @ V.T





if __name__ == "__main__":
	from os import listdir
	from os.path import isfile, join

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
	estimateRelativePose(images[idx1], images[idx2], idx1, idx2) # estimate time 35.433154821395874
	# findCameraIntrinsic()