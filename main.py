from os import listdir
from os.path import isfile, join
from ViewSet import *
from createDepthMap import *
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
v = ViewSet()

# set instrinsics
v.setIntrinsics(matrixList[0])

for i in range(numImages):
	for j in range(i+1, numImages):
		R, t, srcPts, dstPts = estimateRelativePose(images[i], images[j],
		                                            i, j, v.intrinsics)  # estimate time 35.433154821395874
		v.addConnection(i, j, R, t, srcPts, dstPts)

# find the matches
listOfMatches = []
discovered = [dict() for i in range(numImages)]
v.findPointTracks(listOfMatches, discovered)
# find the projections
index = np.random.randint(0, len(listOfMatches))
pointCloud = np.zeros((len(listOfMatches), 6))
for m in range(len(listOfMatches)):
	listOfMatches[m].findProjections(v)
	listOfMatches[m].findBestProjection(v)
	pointCloud[m][0:3] = listOfMatches[m].bestProjection.reshape(3)
	pointCloud[m][3:] = np.array([255., 0., 0.])
finish = time()
print(finish - start)
plot_pointCloud(pointCloud)



