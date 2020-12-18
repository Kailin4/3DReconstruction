'''
Class to store information between all pair of views.
Inspired by: https://www.mathworks.com/help/vision/ref/imageviewset.html
This website had the ImageViewSet class, with a connections table.
Since Matlab is propietary, we could only see the function headers
and reverse engineered the class with our needed functionaly.
'''

from ConnectionsConstants import *
from Match import *


class Connections:
	def __init__(self, numViews):
		'''
		Constructor for Connections objects
		:param numViews: number of views or images
		intializes:
		1. the connections table to store information of
		pairwise matched features
		2. the intrinsics matrix for later computation
		'''
		self.numViews = numViews
		self.connections = [[] for i in range(CONNECTION_ATTRIBUTES)]
		self.intrinsics = None

	def setIntrinsics(self, intrinsics):
		'''
		Sets the intrinsic matrix
		:param intrinsics: camera intrinsics matrix
		:return:
		'''
		self.intrinsics = intrinsics

	def addConnection(self, viewId1, viewId2, relRot, relTrans, srcPts, dstPts):
		'''
		Stores the information the relates the two views
		:param viewId1: first view number
		:param viewId2: second view number
		:param relRot: extrinsics rotation matrix
		:param relTrans: extrinsics translation vector
		:param srcPts: matching 2D features in the first image
		:param dstPts: matching 2D features in the second image
		:return:
		'''
		self.connections[VIEW_ID_1].append(viewId1)
		self.connections[VIEW_ID_2].append(viewId2)
		self.connections[RELATIVE_ROTATION].append(relRot)
		self.connections[RELATIVE_TRANSLATION].append(relTrans)
		self.connections[SOURCE_POINTS].append(srcPts)
		self.connections[DESTINATION_POINTS].append(dstPts)

	def getRelativeTransforms(self, viewId1, viewId2):
		'''
		O(1) access of connections table with triangular
		:param viewId1: first view number
		:param viewId2: second view number
		:return: relative rotation and translation between views
		'''
		index = viewId1 * (self.numViews) + viewId2 - (viewId1+1)*(viewId1+2)//2
		return self.connections[RELATIVE_ROTATION][index], self.connections[RELATIVE_TRANSLATION][index]

	def findMatches(self, listOfMatches, discovered):
		'''
		Finds the matches across all views from pairwise matches of features
		Graph algorithm: Iterate through all edges to find connected components
		:param listOfMatches: empty list to store Match objects
		:param discovered: empty list of dicts [viewid][image pt : match object]
		:return:
		'''
		# for every connection
		for i in range(len(self.connections[VIEW_ID_1])):
			# Get the view id of the points
			viewId1 = self.connections[VIEW_ID_1][i]
			viewId2 = self.connections[VIEW_ID_2][i]
			for j in range(len(self.connections[SOURCE_POINTS][i])):
				# Get the coordinates of the point
				pt1 = tuple(self.connections[SOURCE_POINTS][i][j].astype('int16'))
				pt2 = tuple(self.connections[DESTINATION_POINTS][i][j].astype('int16'))
				# check if the point is in discovered
				# consider the cases:
				# if pt1 is discovered and pt2 is not then add to the existing match
				if pt1 in discovered[viewId1] and pt2 not in discovered[viewId2]:
					match = discovered[viewId1][pt1]
					match.addMatch(viewId2, pt2)
					discovered[viewId2][pt2] = match

				# if pt1 is not discovered and pt2 is discovered add to existing match
				elif pt1 not in discovered[viewId1] and pt2 in discovered[viewId2]:
					match = discovered[viewId2][pt2]
					match.addMatch(viewId1, pt1)
					discovered[viewId1][pt1] = match

				# if pt1 and pt2 are not discovered add both to the same match
				elif pt1 not in discovered[viewId1] and pt2 not in discovered[viewId2]:
					match = Match()
					match.addMatch(viewId1, pt1)
					match.addMatch(viewId2, pt2)
					discovered[viewId1][pt1] = match
					discovered[viewId2][pt2] = match
					listOfMatches.append(match)

	def debugConnections(self):
		'''
		Method to print all connections information.
		:return:
		'''
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
