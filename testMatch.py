'''
File used to test toy examples for
earlier versions of Match class
'''

from Match import *

def testMatchViews():
	m = Match()
	viewId = 0
	pt = np.array([0,0])
	m.addMatch(viewId, pt)
	m.debugViews()
	viewId = 1
	pt = np.array([1, 1])
	m.addMatch(viewId, pt)
	m.debugViews()
	print(m.getViews())

def testMatchProjections():
	m = Match()
	pass

if __name__ == "__main__":
	testMatchViews()