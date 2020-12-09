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




if __name__ == "__main__":
	# testAddOneView()

	# Code to test multiple views (n=2)
	images = [i for i in range(2)]
	R01 = np.array([[0.999404, 0.033737, 0.00734],
					[-0.033737, 0.908992, 0.415446],
					[0.007342, -0.415446, 0.909588]])
	t01 = np.zeros((3,1))
	rotation = [None, R01]
	translation = [None, t01]
	# testAddMultipleViews(images, rotation, translation)
	testAddOneConnection(R01, t01)