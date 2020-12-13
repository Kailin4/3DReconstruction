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
		self.e = 1e-6

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
		listOfViews = self.getViews()
		for i in range(len(listOfViews)):
			for j in range(i, len(listOfViews)):
				viewId1 = listOfViews[i]
				viewId2 = listOfViews[j]

				pt1 = self.views[viewId1]
				pt2 = self.views[viewId2]

				# rot1 = viewSet.views[ABSOLUTE_ROTATION][viewId1]
				# rot2 = viewSet.views[ABSOLUTE_ROTATION][viewId2]
				#
				# trans1 = viewSet.views[ABSOLUTE_TRANSLATION][viewId1]
				# trans2 = viewSet.views[ABSOLUTE_TRANSLATION][viewId2]
				#
				# # -1 due to the opencv implementation
				# proj1 = np.hstack((rot1,-1 * trans1))
				# proj2 = np.hstack((rot2,-1 * trans2))
				projection = self.lounetHiggins(viewSet, viewId1, viewId2, pt1, pt2)
				# projection = cv2.triangulatePoints(proj1, proj2, pt1, pt2)
				self.addProjection(viewSet, viewId1, viewId2, projection)

	def longuetHiggins(self, viewSet, viewId1, viewId2, y, y_prime):
		relRot, relTrans = viewSet.getRelativeTransforms(viewId1, viewId2)
		r1, r2, r3 = relRot[0], relRot[1], relRot[2]
		fac = (r1 - y_prime[0] * r3)
		# Slight modification to avoid division by zero
		z1 = (fac @ relTrans) / ((fac @ y) + self.e)
		fac = (r2 - y_prime[1] * r3)
		z2 = (fac @ relTrans) / ((fac @ y) + self.e)
		# Take the average of the two estimates
		z = np.avg(z1, z2)[0]
		proj1 = np.zeros(3)


		return None

	def findOptimalProjection(self, viewSet):
		listOfViews = self.getViews()
		listOfProjections = self.getProjections()
		# reproject the projection on each of the views
		smallestError = np.inf
		bestProj = None
		for projection in listOfProjections:
			error = 0
			for viewId in listOfViews:
				pt = self.views[viewId]
				rot = viewSet.views[ABSOLUTE_ROTATION][viewId]
				trans = viewSet.views[ABSOLUTE_TRANSLATION][viewId]
				proj = np.hstack((rot, -1 * trans))
				reproj = proj @ projection
				# make 2d point
				reproj = reproj / reproj[2,0]
				reproj = reproj[0:2]
				reproj = reproj.reshape(2)
				error += np.linalg.norm(reproj - pt)
			if error < smallestError:
				bestProj = projection
		self.bestProjection = bestProj



