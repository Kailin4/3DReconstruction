'''
Script used to run the point cloud pipeline.
Inspired by: https://www.mathworks.com/help/vision/ug/structure-from-motion-from-multiple-views.html
We drew inspiration from the high-level procedure and function calls
as we developed our python pipeline.
'''

from os import listdir
from os.path import isfile, join
from Connections import *
from helpPointCloud import *
from time import time

start = time()
# set the images directory for semper dataset
imgDir = "images/semper/"
outputDir = "output/temp/"

# get list of file names
files = sorted([f for f in listdir(imgDir) if isfile(join(imgDir, f))])
# get list of camera matrix file names
cameraFiles = [imgDir + f for f in files if "camera" in f]
# get list of image file names
imgFiles = [imgDir + f for f in files if "ppm" in f and "camera" not in f and "3D" not in f]
# convert ppm to numpy array
images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in imgFiles]
# images = images[0:2]
numImages = len(images)
# load the camera matrices
matrixFiles = [open(f) for f in cameraFiles]
matrixList = [np.zeros((3, 3)) for f in cameraFiles]
for i in range(len(matrixFiles)):
	for j in range(3):
		row = matrixFiles[i].readline().split()
		for k in range(3):
			matrixList[i][j, k] = np.float(row[k])

# instantiate viewSet object
c = Connections(len(images))

# set instrinsics
c.setIntrinsics(matrixList[0])

# fill connection table
for i in range(numImages):
	for j in range(i+1, numImages):
		R, t, srcPts, dstPts = estimateRelativeExtrinsics(images[i], images[j],
		                                                  i, j, c.intrinsics)  # estimate time 35.433154821395874
		c.addConnection(i, j, R, t, srcPts, dstPts)

# find the matches
listOfMatches = []
discovered = [dict() for i in range(numImages)]
c.findMatches(listOfMatches, discovered)
# find the 3D points
index = np.random.randint(0, len(listOfMatches))
compute3DPoints(listOfMatches, c, False)
filterRelativeProjections(listOfMatches)
compute3DPoints(listOfMatches, c, True)
listOfMatches = filterWorldPoints(listOfMatches)
pointCloud = createPointCloud(listOfMatches)
finish = time()
print(finish - start)
plot_pointCloud(pointCloud)



