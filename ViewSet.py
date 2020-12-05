from ViewSetConstants import *
import numpy as np
class ViewSet:
	def __init__(self):
		self.numViews = 0
		self.numConnections = 0
		self.views = [[] for i in range(VIEW_ATTRIBUTES)]
		self.connections = [[] for i in range(CONNECTION_ATTRIBUTES)]

	def addView(self, viewId, relRot, relTrans, desc, kp):
		self.views[VIEW_ID].append(viewId)
		self.views[DESCRIPTORS].append(desc)
		self.views[KEYPOINTS].append(kp)
		self.addAbsolutePose(viewId, relRot, relTrans)

	def addConnection(self, viewId1, viewId2, relRot, relTrans, matches):
		self.connections[VIEW_ID_1].append(viewId1)
		self.connections[VIEW_ID_2].append(viewId2)
		self.connections[RELATIVE_ROTATION].append(relRot)
		self.connections[RELATIVE_TRANSLATION].append(relTrans)
		self.connections[MATCHES].append(matches)

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




