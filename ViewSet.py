from ViewSetConstants import *
import numpy as np
class ViewSet:
	def __init__(self):
		self.numViews = 0
		self.numConnections = 0
		self.views = [[] for i in range(VIEW_ATTRIBUTES)]
		self.connections = [[] for i in range(CONNECTION_ATTRIBUTES)]

	def addView(self, viewId, name, value):
		self.views[VIEW_ID].append(viewId)
		self.views[NAME].append(name)
		self.views[POINTS].append(value)

	def addConnection(self, viewId1, viewId2, relativeRotation, relativeTranslation, informationMatrix, matches):
		self.connections[VIEW_ID_1].append(viewId1)
		self.connections[VIEW_ID_2].append(viewId2)
		self.connections[RELATIVE_ROTATION].append(relativeRotation)
		self.connections[RELATIVE_TRANSLATION].append(relativeTranslation)
		self.addAbsolutePose(viewId1, viewId2)
		self.connections[MATCHES].append(matches)

	def addAbsolutePose(self, viewId1):
		# Note about opencv2 translation is cam2 to cam1 in cam2 frame

		# Rotation from cam2 to cam1
		R2To1 = np.linalg.inv(self.connections[RELATIVE_ROTATION][-1])
		# Translation from cam2 to cam1 (now in cam1 frame)
		t2To1 = R2To1 @ self.connections[RELATIVE_TRANSLATION][-1]
		assert self.connections[VIEW_ID_1][viewId1-1] == 0
		assert self.connections[VIEW_ID_2][viewId1-1] == viewId1
		# Rotation from cam1 to absolute
		R1ToAbsolute = np.linalg.inv(self.connections[RELATIVE_TRANSLATION][viewId1-1])
		# Translation from cam1 to absolute in (absolute frame)
		t1ToAbsolute = R1ToAbsolute @ self.connections[RELATIVE_TRANSLATION][viewId1-1]
		absoluteRotation = R2To1 @ R1ToAbsolute
		absoluteTranslation = R1ToAbsolute @ t2To1 + t1ToAbsolute
		self.connections[ABSOLUTE_ROTATION].append(absoluteRotation)
		self.connections[ABSOLUTE_TRANSLATION].append(absoluteTranslation)




