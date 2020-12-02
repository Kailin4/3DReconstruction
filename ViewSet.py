from ViewSetConstants import *

class ViewSet:
	def __init__(self):
		self.numViews = 0
		self.numConnections = 0
		self.views = [[] for i in range(VIEW_ATTRIBUTES)]
		self.connections = [[] for i in range(CONNECTION_ATTRIBUTES)]

	def addView(self, viewId, absoluteRotation, absoluteTranslation, name, value):
		self.views[VIEW_ID].append(viewId)
		self.views[ABSOLUTE_ROTATION].append(absoluteRotation)
		self.views[ABSOLUTE_TRANSLATION].append(absoluteTranslation)
		self.views[NAME].append(name)
		self.views[POINTS].append(value)

	def addConnection(self, viewId1, viewId2, relativeRotation, relativeTranslation, informationMatrix, matches):
		self.connections[VIEW_ID_1].append(viewId1)
		self.connections[VIEW_ID_2].append(viewId2)
		self.connections[RELATIVE_ROTATION].append(relativeRotation)
		self.connections[RELATIVE_TRANSLATION].append(relativeTranslation)
		self.connections[INFORMATION_MATRIX].append(informationMatrix)
		self.connections[MATCHES].append(matches)

