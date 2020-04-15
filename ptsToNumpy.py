"""
Written by Ruiqi Chen
April 1, 2019
This code takes in a regularized point cloud file (.pts; possibly with headers representing spacing) and converts to numpy array where array[i, j, k] is a vector (1D array) with values (x, y, z, r, g, b, a, c)
where x, y, z are coordinates; r, g, b, a are color intensity values between 0-255, and c takes the value of -1 (cracked), 0 (unknown), 1 cracked)
"""
import numpy as np

def ptsToNumpy(ptsFile, spacing=None, coordinateColumns=(0, 1, 2), colorColumns=(3, 4, 5, 6)):
	# each line in the pts file should be a comma separated list; nominally x y z i r g b
	# there's some room for customization if you specify which are the coordinate (xyz) columns and which are the color (irgb) columns
	# if spacing=None, the first three lines in the pts file MUST be three numbers representing grid spacing in x y z directions
	# otherwise, spacing should be provided as a tuple; e.g. spacing=(0.001, 0.001, 0.002)

	# determine spacing, if not given
	if spacing == None:
		f = open(ptsFile, "r")
		lineNum = 0
		for line in f:
			if lineNum == 0:
				dx = float(line)
			elif lineNum == 1:
				dy = float(line)
			elif lineNum == 2:
				dz = float(line)
			else:
				break
			lineNum += 1
		f.close()
		spacing = (dx, dy, dz)
		data = np.loadtxt(ptsFile, delimiter=",", skiprows=3, usecols=coordinateColumns + colorColumns)
	else:
		data = np.loadtxt(ptsFile, delimiter=",", usecols=coordinateColumns + colorColumns)

	# find extents
	minX = min(data[:, 0])
	maxX = max(data[:, 0])
	minY = min(data[:, 1])
	maxY = max(data[:, 1])
	minZ = min(data[:, 2])
	maxZ = max(data[:, 2])
	numX = int((maxX - minX)/spacing[0] + 1)
	numY = int((maxY - minY)/spacing[1] + 1)
	numZ = int((maxZ - minZ)/spacing[2] + 1)

	# initialize resulting numpy array
	result = np.zeros((numX, numY, numZ, len(coordinateColumns) + len(colorColumns) + 2))

	# loop through data and put into result
	for pt in range(data.shape[0]):
		i = int((data[pt, 0] - minX)/spacing[0])
		j = int((data[pt, 1] - minY)/spacing[1])
		k = int((data[pt, 2] - minZ)/spacing[2])
		result[i, j, k] = np.concatenate(data[pt], 0, 0)

	return result