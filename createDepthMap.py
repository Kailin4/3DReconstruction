import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from time import time

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

def estimateRelativePose(img1, img2):
	pass


if __name__ == "__main__":
	pass