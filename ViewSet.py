from ViewSetConstants import *
from Match import *

class ViewSet:
	def __init__(self):
		self.numViews = 0
		self.numConnections = 0
		self.views = [[] for i in range(VIEW_ATTRIBUTES)]
		self.connections = [[] for i in range(CONNECTION_ATTRIBUTES)]

	def addView(self, viewId, relRot, relTrans):# desc, kp):
		self.numViews += 1
		self.views[VIEW_ID].append(viewId)
		# self.views[DESCRIPTORS].append(desc)
		# self.views[KEYPOINTS].append(kp)
		self.addAbsolutePose(viewId, relRot, relTrans)

	def addAbsolutePose(self, viewId, relRot, relTrans):
		# Note about opencv2 translation is cam2 to cam1 in cam2 frame.
		# In our algorithm we have a fundamental assumption, we add connections
		# in the following order: 0-1, 1-2, ..., (n-2)-(n-1).
		# That means there are n-1 stored connections for n images (zero-indexed).
		if viewId == 0:
			self.views[ABSOLUTE_ROTATION].append(np.eye(3))
			self.views[ABSOLUTE_TRANSLATION].append(np.zeros((3,1)))
			return

		# Another note cv2.triangulatePoints projects from world coordinates to the image coordinates
		# So ABSOLUTE ROTATION stores absolute to image
		absR = self.views[ABSOLUTE_ROTATION][viewId-1] @ relRot
		# Keep in mind cv2 x2 = R(x1) + t
		absT = absR @ relTrans + self.views[ABSOLUTE_TRANSLATION][viewId-1]
		# later have to multiply by -1
		self.views[ABSOLUTE_ROTATION].append(absR)
		self.views[ABSOLUTE_TRANSLATION].append(absT)

	def addConnection(self, viewId1, viewId2, relRot, relTrans, srcPts, dstPts):
		self.connections[VIEW_ID_1].append(viewId1)
		self.connections[VIEW_ID_2].append(viewId2)
		self.connections[RELATIVE_ROTATION].append(relRot)
		self.connections[RELATIVE_TRANSLATION].append(relTrans)
		self.connections[SOURCE_POINTS].append(srcPts)
		self.connections[DESTINATION_POINTS].append(dstPts)

	def getRelativeTransforms(self, viewId1, viewId2):
		index = viewId1 * (self.numViews) + viewId2 - (viewId1+1)*(viewId1+2)//2
		return self.connections[RELATIVE_ROTATION][index], self.connections[RELATIVE_TRANSLATION][index]

	def findPointTracks(self, listOfMatches, discovered):
		# for every connection
		for i in range(len(self.connections[VIEW_ID_1])):
			# Get the view id of the points
			viewId1 = self.connections[VIEW_ID_1][i]
			viewId2 = self.connections[VIEW_ID_2][i]
			for j in range(len(self.connections[SOURCE_POINTS][i])):
				# Get the coordinates of the point
				pt1 = tuple(self.connections[SOURCE_POINTS][i][j]) # maybe convert to int tuple here
				pt2 = tuple(self.connections[DESTINATION_POINTS][i][j]) # maybe convert to int tuple
				# pt1 = tuple(self.connections[SOURCE_POINTS][i][j].astype('int16')) # maybe convert to int tuple here
				# pt2 = tuple(self.connections[DESTINATION_POINTS][i][j].astype('int64')) # maybe convert to int tuple
				# check if the point is in discovered
				# consider the cases:
				# if pt1 and pt2 are discovered do nothing
				# if pt1 is discovered and pt2 is not then add to the existing match
				# if pt1 is not discovered and pt2 is is discovered add to existing match
				# if pt1 and pt2 are not discovered add both to the same match
				# Note pt1 must be a tuple (can also convert to int)
				if pt1 in discovered[viewId1] and pt2 not in discovered[viewId2]:
					match = discovered[viewId1][pt1]
					match.addMatch(viewId2, pt2)
					discovered[viewId2][pt2] = match

				if pt1 not in discovered[viewId1] and pt2 in discovered[viewId2]:
					match = discovered[viewId2][pt2]
					match.addMatch(viewId1, pt1)
					discovered[viewId1][pt1] = match

				if pt1 not in discovered[viewId1] and pt2 not in discovered[viewId2]:
					match = Match()
					match.addMatch(viewId1, pt1)
					match.addMatch(viewId2, pt2)
					discovered[viewId1][pt1] = match
					discovered[viewId2][pt2] = match
					listOfMatches.append(match)

	def debugViews(self):
		print("numViews: ", self.numViews)
		for key, val in viewAttributesDict.items():
			print(key, self.views[val])

	def debugConnections(self):
		for key, val in connectionAttributesDict.items():
			print(key)
			try:
				iterator = iter(self.connections[val])
			except TypeError:
			# not iterable
				print(self.connections[val])
			else:
				for item in self.connections[val]:
					print(item)
