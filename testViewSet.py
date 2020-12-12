from ViewSet import *

def testAddOneView():
	v = ViewSet()
	v.addView(0, None, None, None, None)
	v.debugViews()

def testAddMultipleViews(images, rotation, translation):
	# Just test the rotation and translation
	v = ViewSet()
	v.addView(0, None, None, None, None)
	for i in range(1, len(images)):
		v.addView(i, rotation[i], translation[i], None, None)
	v.debugViews()

def testAddOneConnection(R01, t01):
	v = ViewSet()
	v.addConnection(0,1, R01, t01, None, None)
	v.debugConnections()

def testFindPointTracksTwo():
	v = ViewSet()
	srcPts = np.array([[1,1],
	                   [3,3],
	                   [5,5]])
	dstPts = np.array([[2,2],
	                     [4,4],
	                     [6,6]])
	v.addConnection(0, 1, None, None, srcPts, dstPts)
	v.debugConnections()
	listOfMatches = []
	discovered = [dict(),dict()]
	v.findPointTracks(listOfMatches, discovered)
	[match.debugViews() for match in listOfMatches]

def testFindPointTracksThree():
	v = ViewSet()
	srcPts01 = np.array([[1,1],
	                   [4,4]])
	dstPts01 = np.array([[2,2],
	                   [5,5]])
	v.addConnection(0, 1, None, None, srcPts01, dstPts01)
	srcPts02 = np.array([[8,8]])
	dstPts02 = np.array([[9,9]])
	v.addConnection(0, 2, None, None, srcPts02, dstPts02)
	srcPts12 = np.array([[2,2],
	                     [6,6]])
	dstPts12 = np.array([[3,3],
	                     [7,7]])
	v.addConnection(1, 2, None, None, srcPts12, dstPts12)
	# v.debugConnections()
	listOfMatches = []
	discovered = [dict(),dict(), dict()]
	v.findPointTracks(listOfMatches, discovered)
	[match.debugViews() for match in listOfMatches]

def testFindPointTracksFour():
	v = ViewSet()
	srcPts01 = np.array([[1,1],
	                     [8,8],
	                     [17,17]])
	dstPts01 = np.array([[2,2],
	                     [9,9],
	                     [18,18]])
	v.addConnection(0, 1, None, None, srcPts01, dstPts01)
	srcPts02 = np.array([[5,5],
	                     [19,19]])
	dstPts02 = np.array([[7,7],
	                     [20,20]])
	v.addConnection(0, 2, None, None, srcPts02, dstPts02)
	srcPts03 = np.array([[8,8],
	                     [11,11]])
	dstPts03 = np.array([[10,10],
	                     [13,13]])
	v.addConnection(0, 3, None, None, srcPts03, dstPts03)
	srcPts12 = np.array([[2,2],
	                     [6,6],
	                     [14,14],
	                     [21,21]])
	dstPts12 = np.array([[3,3],
	                     [7,7],
	                     [15,15],
	                     [22,22]])
	v.addConnection(1, 2, None, None, srcPts12, dstPts12)

	srcPts13 = np.array([[14,14],
	                     [23,23]])
	dstPts13 = np.array([[16,16],
	                     [24,24]])
	v.addConnection(1, 3, None, None, srcPts13, dstPts13)

	srcPts23 = np.array([[3,3],
	                     [12,12],
	                     [25,25]])
	dstPts23 = np.array([[4,4],
	                     [13,13],
	                     [26,26]])
	v.addConnection(2, 3, None, None, srcPts23, dstPts23)


	# v.debugConnections()
	listOfMatches = []
	discovered = [dict(),dict(), dict(), dict()]
	v.findPointTracks(listOfMatches, discovered)
	[match.debugViews() for match in listOfMatches]

if __name__ == "__main__":
	# testAddOneView()
	'''
	# Code to test adding multiple views (n=2)
	images = [i for i in range(2)]
	R01 = np.array([[0.999404, 0.033737, 0.00734],
					[-0.033737, 0.908992, 0.415446],
					[0.007342, -0.415446, 0.909588]])
	t01 = np.zeros((3,1))
	rotation = [None, R01]
	translation = [None, t01]
	testAddMultipleViews(images, rotation, translation)
	'''
	'''
	# Code to test adding one connection
	testAddOneConnection(R01, t01)
	'''
	'''
	# Code to test find matches in one pair of views
	testFindPointTracksTwo()
	'''
	'''
	# Code to test find matches in three views
	testFindPointTracksThree()
	'''
	'''
	Code to test find matches in three views
	testFindPointTracksFour()
	# '''
