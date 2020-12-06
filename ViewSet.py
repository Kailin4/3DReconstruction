from ViewSetConstants import *
import numpy as np

class Match:
	def __init__(self, viewId1, viewId2, pt1, pt2):
		self.images = dict()
		self.images[viewId1] = pt1
		self.images[viewId2] = pt2
	def addMatch(self, viewId, pt):
		self.images[viewId] = pt


class ViewSet:
	def __init__(self, numViews):
		self.numViews = numViews
		self.numConnections = 0
		self.views = [[] for i in range(VIEW_ATTRIBUTES)]
		self.connections = [[] for i in range(CONNECTION_ATTRIBUTES)]
		# listofMatches stores instances of Match (which map image # to location on image plane)
		self.listOfMatches = []
		# self.discovered is an list of dictionaries which indicate if a point
		# has been added to the listOfMatches before.
		# The key is a tuple for the coordinate, and the value is the reference of the match instance.
		self.discovered = [dict() for i in range(numViews)]


	def addView(self, viewId, relRot, relTrans, desc, kp):
		self.views[VIEW_ID].append(viewId)
		self.views[DESCRIPTORS].append(desc)
		self.views[KEYPOINTS].append(kp)
		self.addAbsolutePose(viewId, relRot, relTrans)

	def addConnection(self, viewId1, viewId2, relRot, relTrans, srcPts, dstPts):
		self.connections[VIEW_ID_1].append(viewId1)
		self.connections[VIEW_ID_2].append(viewId2)
		self.connections[RELATIVE_ROTATION].append(relRot)
		self.connections[RELATIVE_TRANSLATION].append(relTrans)
		self.connections[SOURCE_POINTS].append(srcPts)
		self.connections[DESTINATION_POINTS].append(dstPts)

	def addAbsolutePose(self, viewId, relRot, relTrans):
		# Note about opencv2 translation is cam2 to cam1 in cam2 frame.
		# In our algorithm we have a fundamental assumption, we add connections
		# in the following order: 0-1, 1-2, ..., (n-2)-(n-1).
		# That means there are n-1 stored connections for n images (zero-indexed).

		if viewId == 0:
			self.views[ABSOLUTE_ROTATION].append(np.eye(3))
			self.views[ABSOLUTE_TRANSLATION].append(np.zeros((3,1)))
			return

		invRot = np.linalg.inv(relRot)

		absR = self.views[ABSOLUTE_ROTATION][viewId-1] @ invRot
		absT = absR @ relTrans + self.views[ABSOLUTE_TRANSLATION][viewId-1]

		#
		#
		# # Rotation from cam2 to cam1
		# R1ToAbsolute = np.eye(3)
		# t1ToAbsolute =
		#
		#
		# # Rotation from cam2 to cam1 (now in cam1 frame), added in addConnection
		# R2To1 = np.linalg.inv(self.connections[RELATIVE_ROTATION][-1])
		# # Translation from cam2 to cam1 (now in cam1 frame), added in addConnection
		# t2To1 = self.connections[RELATIVE_TRANSLATION][-1]
		#
		#
		#
		#
		#
		# for i in range(viewId1-1, -1, -1):
		# 	# Rotation from cam2 to cam1 (now in cam1 frame)
		# 	R2To1 = np.linalg.inv(self.connections[RELATIVE_ROTATION][i])
		# 	# Translation from cam2 to cam1 (now in cam1 frame)
		# 	t2To1 = self.connections[RELATIVE_TRANSLATION][i]
		# 	# Rotation from cam1 to absolute
		# 	R1ToAbsolute = R1ToAbsolute @ R2To1
		# 	t1ToAbsolute = R2To1 @ (t1ToAbsolute + t2To1)
		# # Translation from cam1 to absolute in (absolute frame)
		# t1ToAbsolute = R1ToAbsolute @ self.connections[RELATIVE_TRANSLATION][viewId1-1]
		# absoluteRotation = R2To1 @ R1ToAbsolute
		# absoluteTranslation = R1ToAbsolute @ t2To1 + t1ToAbsolute

		self.connections[ABSOLUTE_ROTATION].append(absR)
		self.connections[ABSOLUTE_TRANSLATION].append(absT)

	def findPointTracks(self):
		# for every connection
		for i in range(len(self.connections[VIEW_ID_1])):
			# Get the view id of the points
			viewId1 = self.connections[VIEW_ID_1][i]
			viewId2 = self.connections[VIEW_ID_2][i]
			for j in range(len(self.connections[SOURCE_POINTS][i])):
				# Get the coordinates of the point
				pt1 = self.connections[SOURCE_POINTS][i][j] # maybe convert to int tuple here
				pt2 = self.connections[DESTINATION_POINTS][i][j] # maybe convert to int tuple
				# check if the point is in discovered
				# consider the cases:
				# if pt1 and pt2 are discovered do nothing
				# if pt1 is discovered and pt2 is not then add to the existing match
				# if pt1 is not discovered and pt2 is is discovered add to existing match
				# if pt1 and pt2 are not discovered add both to the same match
				# Note pt1 must be a tuple (can also convert to int)
				if pt1 in self.discovered[viewId1] and pt2 not in self.discovered[viewId2]:
					match = self.discovered[viewId1][pt1]
					match.addMatch(viewId2, pt2)
					self.discovered[viewId2][pt2] = match

				if pt1 not in self.discovered[viewId2] and pt2 in self.discovered[viewId1]:
					match = self.discovered[viewId2][pt2]
					match.addMatch(viewId1, pt1)
					self.discovered[viewId1][pt1] = match

				if pt1 not in self.discovered[viewId1] and pt2 not in self.discovered[viewId2]:
					match = Match(viewId1, viewId2, pt1, pt2)
					self.discovered[viewId1][pt1] = match
					self.discovered[viewId2][pt2] = match
					self.listOfMatches.append(match)





if __name__ == "__main__":
	a = True
	if a:
		match = 5
	print(match)
