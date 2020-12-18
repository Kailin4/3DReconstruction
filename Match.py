'''
Class to store the relationship between
a single 2D feature with multiple views and
its associated 3D representations
'''

from ConnectionsConstants import *
import numpy as np
import cv2


class Match:
	def __init__(self):
		'''
		Constructor for Connections objects
		:param numViews: number of views or images
		intializes:
		1. the views dictionary which maps a ViewId to a 2D feature
		2. the camera points dictionary which maps a pair of views to a camera space point.
		'''
		self.views = dict()
		# viewId: pt tuple
		self.cameraSpacePoints = dict()
		# tuple (2 view ids):
		self.worldPoint = None

	def debugViews(self): print(self.views)

	def debugCameraPoints(self): print(self.cameraSpacePoints)

	def debugWorldPoint(self): print(self.worldPoint)

	def addMatch(self, viewId, pt):
		self.views[viewId] = pt

	def getViews(self):
		return self.views.keys()

	def getCameraSpacePoints(self):
		return self.cameraSpacePoints.values()

	def addCameraSpacePoint(self, viewId1, viewId2, cameraSpacePoint):
		self.cameraSpacePoints[(viewId1, viewId2)] = cameraSpacePoint

	def findCameraSpacePoints(self, connections):
		'''
		Find the all the corresponding points in camera space for
		a matching feature
		:param connections: Connections object
		:return:
		'''
		# get list of views
		listOfViews = sorted(list(self.getViews()))
		for i in range(len(listOfViews)):
			for j in range(i+1, len(listOfViews)):
				# create all posible pairs of views
				viewId1 = listOfViews[i]
				viewId2 = listOfViews[j]

				pt1 = self.views[viewId1]
				pt2 = self.views[viewId2]
				# find the 3D coordinates
				self.longuetHiggins(connections, viewId1, viewId2, pt1, pt2)

	def longuetHiggins(self, connections, viewId1, viewId2, pt1, pt2):
		'''
		Calculates the camera space point and assigns it to CameraSpacePoint
		:param connections: connections object
		:param viewId1: first view number
		:param viewId2: second view number
		:param pt1: first view coordinates of feature
		:param pt2: second view coordinates of corresponding feature
		:return:
		'''
		# y and y' is a homogenous vector with image
		# form homogenous coordinates i
		y = np.ones(3)
		y[0:2] = pt1
		y_prime = np.ones(3)
		y_prime[0:2] = pt2
		# get extrinsics
		relRot, relTrans = connections.getRelativeTransforms(viewId1, viewId2)
		r1, r2, r3 = relRot[0], relRot[1], relRot[2]
		fac = (r1 - y_prime[0] * r3)
		# Use Longuet Higgins method to solve for z coodrdinate
		# Slight modification to avoid division by zero
		# Symmetric equations, we can get two estimates
		z1 = ((fac @ relTrans) / ((fac @ np.array(y).reshape(3,1)) + E))[0]
		fac = (r2 - y_prime[1] * r3)
		z2 = ((fac @ relTrans) / ((fac @ np.array(y).reshape(3,1)) + E))[0]
		# Take the average of the two estimates
		z = (z1 + z2) / 2
		# Compute the camera space point
		cameraSpacePoint = np.ones(3)
		cameraSpacePoint[0:2] = pt1
		cameraSpacePoint *= z
		cameraSpacePoint = cameraSpacePoint.reshape(3,1)
		# 3D coordinates in the first viewId1 space
		self.addCameraSpacePoint(viewId1, viewId2, cameraSpacePoint)

	def filterCameraSpacePoints(self, lowerBound, upperBound):
		'''
		Filters Camera Space Points whose norms are outliers
		:param lowerBound: 1st quartile - 1.5 * Interquartile Range
		:param upperBound: 3rd quartile + 1.5 * Interquartile Range
		:return:
		'''
		# for each camera space point
		for views in self.cameraSpacePoints.keys():
			p = self.cameraSpacePoints[views]
			# get the norm
			normP = np.linalg.norm(p)
			# set the point to None if it is an outlier
			if normP > upperBound or normP < lowerBound:
				self.cameraSpacePoints[views] = None

	def findWorldPoint(self, viewSet):
		'''
		Find the Camera Space Point that minimizes reprojection error.
		:param viewSet:
		:return:
		'''
		listOfViews = sorted(list(self.getViews()))
		minError = np.inf
		bestProj = None
		for views, cameraSpacePt in self.cameraSpacePoints.items():
			# if it is outlier we ignore this point
			if cameraSpacePt is None: continue
			error = 0
			viewId1 = views[0]
			for viewIdTarget in listOfViews: # for each view
				# get the 2D coordinates of the feature
				targetPt = self.views[viewIdTarget]
				# get the extrinsics
				relRot, relTrans = viewSet.getRelativeTransforms(viewId1, viewIdTarget)
				# reproject the camera space point on each of the views
				reproj = (viewSet.intrinsics @ (relRot @ cameraSpacePt + relTrans)).reshape(3)
				reproj /= reproj[2]
				reproj = reproj[0:2]
				# find the error
				error += np.linalg.norm(reproj - targetPt)
			if error < minError:
				# convert to world space (view 0)
				if viewId1 != 0:
					relRot, relTrans = viewSet.getRelativeTransforms(0, viewId1)
					bestProj = np.linalg.inv(relRot) @ (cameraSpacePt - relTrans)
				else: bestProj = cameraSpacePt
		# set the world point
		self.worldPoint = bestProj
