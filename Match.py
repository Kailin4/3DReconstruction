class Match:
	def __init__(self):
		self.views = dict()
		# viewId: pt tuple
		self.projections = dict()
		# tuple (2 view ids):
	def addMatch(self, viewId, pt):
		self.views[viewId] = pt
	def getViews(self):
		return self.views.keys()
	def addProjection(self, viewId1, viewId2, projection):
		self.projections[(viewId1, viewId2)] = projection
