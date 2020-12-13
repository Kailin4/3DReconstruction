from os import listdir
from os.path import isfile, join
from createDepthMap import *
from ViewSet import *



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
# load the camera matrices
matrixFiles = [open(f) for f in cameraFiles]
matrixList = [np.zeros((3, 3)) for f in cameraFiles]
for i in range(len(matrixFiles)):
	for j in range(3):
		row = matrixFiles[i].readline().split()
		for k in range(3):
			matrixList[i][j, k] = np.float(row[k])
idx1, idx2 = 0, 1
R, t = estimateRelativePose(images[idx1], images[idx2], idx1, idx2)  # estimate time 35.433154821395874
# print(R)
# print(t)
