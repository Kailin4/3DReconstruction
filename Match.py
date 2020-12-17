from ViewSetConstants import *
import numpy as np
import cv2


class Match:
	def __init__(self):
		self.views = dict()
		# viewId: pt tuple
		self.projections = dict()
		# tuple (2 view ids):
		self.bestProjection = None
		self.e = [1e-6]

	def debugViews(self): print(self.views)

	def debugProjections(self): print(self.projections)

	def debugBestProjection(self): print(self.bestProjection)

	def addMatch(self, viewId, pt):
		self.views[viewId] = pt

	def getViews(self):
		return self.views.keys()

	def getProjections(self):
		return self.projections.values()

	def addProjection(self, viewId1, viewId2, projection):
		self.projections[(viewId1, viewId2)] = projection

	def findProjections(self, viewSet):
		listOfViews = sorted(list(self.getViews()))
		for i in range(len(listOfViews)):
			for j in range(i+1, len(listOfViews)):
				viewId1 = listOfViews[i]
				viewId2 = listOfViews[j]

				pt1 = self.views[viewId1]
				pt2 = self.views[viewId2]

				self.longuetHiggins(viewSet, viewId1, viewId2, pt1, pt2)
				# projection = cv2.triangulatePoints(proj1, proj2, pt1, pt2)

	def longuetHiggins(self, viewSet, viewId1, viewId2, pt1, pt2):
		# y and y' is (2, tuple)
		y = np.ones(3)
		y[0:2] = pt1
		y_prime = np.ones(3)
		y_prime[0:2] = pt2
		relRot, relTrans = viewSet.getRelativeTransforms(viewId1, viewId2)
		r1, r2, r3 = relRot[0], relRot[1], relRot[2]
		fac = (r1 - y_prime[0] * r3)
		# Slight modification to avoid division by zero
		z1 = ((fac @ relTrans) / ((fac @ np.array(y).reshape(3,1)) + self.e))[0]
		fac = (r2 - y_prime[1] * r3)
		z2 = ((fac @ relTrans) / ((fac @ np.array(y).reshape(3,1)) + self.e))[0]
		# Take the average of the two estimates
		z = (z1 + z2) / 2
		projection = np.ones(3)
		projection[0:2] = pt1
		projection *= z
		projection = projection.reshape(3,1)# / np.linalg.norm(projection)
		# 3D coordinates in the first viewId1 space
		self.addProjection(viewId1, viewId2, projection)

	def filterProjections(self, lowerBound, upperBound):
		for views in self.projections.keys():
			p = self.projections[views]
			normP = np.linalg.norm(p)
			if normP > upperBound or normP < lowerBound:
				self.projections[views] = None

	def findBestProjection(self, viewSet):
		listOfViews = sorted(list(self.getViews()))
		# reproject the projection on each of the views
		minError = np.inf
		bestProj = None
		for views, pt in self.projections.items():
			if pt is None: continue
			error = 0
			viewId1 = views[0]
			for viewIdTarget in listOfViews:
				if viewId1 != viewIdTarget:
					targetPt = self.views[viewIdTarget]
					relRot, relTrans = viewSet.getRelativeTransforms(viewId1, viewIdTarget)
					reproj = (viewSet.intrinsics @ (relRot @ pt + relTrans)).reshape(3)
					reproj /= reproj[2]
					reproj = reproj[0:2]
					error += np.linalg.norm(reproj - targetPt)
			if error < minError:
				# convert to world space
				if viewId1 != 0:
					relRot, relTrans = viewSet.getRelativeTransforms(0, viewId1)
					bestProj = np.linalg.inv(relRot) @ (pt - relTrans)
				else: bestProj = pt
				# bestProj /= np.linalg.norm(bestProj)
		self.bestProjection = bestProj
