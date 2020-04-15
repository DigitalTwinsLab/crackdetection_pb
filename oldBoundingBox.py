"""
Written by Ruiqi Chen
February 28, 2019
This code takes in a point cloud and finds a minimum bounding box using SVD.
It then centers and rotates the original point cloud to be aligned with Cartesian X, Y, and Z axes.
For efficiency, by default, point sampling is enabled. This can be disabled with the appropriate input arguments.
Methods generateDummyData and demo are provided for testing
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generateDummyData(rx, ry, rz, length, width, height, numTotalPoints=1000, tol=1e-6, mode="rad", plot=False):
    # this generates numTotalPoints in the box bounded by length x width x height and rotated by angles rx, ry, rz in that order
    # angles are assumed to be in degree (mode="rad"); to specify input as radians, use (mode="deg")
    if mode == "deg":
        rx = rx*numpy.pi/180
        ry = ry*numpy.pi/180
        rz = rz*numpy.pi/180
    elif mode == "rad":
        pass
    else:
        raise Exception("Parameter mode must be either \"deg\" or \"rad\"!")

    # generate uniformly sampled points in Cartesian frame first
    points = np.random.random_sample((3, numTotalPoints))
    points[0, :] *= length
    points[1, :] *= width
    points[2, :] *= height
    # rotate to coordinate system defined by input rotations
    rotx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    roty = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rotz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rotpoints = rotz @ roty @ rotx @ points
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(rotpoints[0, :], rotpoints[1, :], rotpoints[2, :])
        ax.set_title("Randomly Generated Point Cloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    return rotpoints

def findCentroid(data):
    # data is an n x 3 or 3 x n numpy matrix containing all x, y, z coordinate of points
    if data.shape[0] == 3:
        return np.mean(data, axis=1)
    elif data.shape[1] == 3:
        return np.mean(data, axis=0)
    else:
        raise Exception("Input data must be of dimensions n x 3 or 3 x n!")

def boundingBox(data, plot=False, returnRotatedData=True, sampleSize=100):
    # data is an n x 3 or 3 x n numpy matrix containing all x, y, z coordinate of points
    # for efficiency, sampling is enabled by default. To disable sampling, change sampleSize to None.
    # if data contains fewer points than the samplingSize, no sampling will take place.
    if data.shape[0] == 3:
        data = np.transpose(data)
    centroid = findCentroid(data)
    shiftedData = data - numpy.matlib.repmat(centroid, data.shape[0], 1)
    if sampleSize == None or sampleSize >= data.shape[0]:
        u, s, vh = np.linalg.svd(shiftedData, full_matrices=False) # rows of vh contain the principal axes
        rotationMatrix = np.transpose(vh)
    else:
        randomSampleIndices = np.random.choice(data.shape[0], size=sampleSize, replace=False)
        dataSample = data[randomSampleIndices, :]
        shiftedDataSample = dataSample - numpy.matlib.repmat(centroid, dataSample.shape[0], 1)
        u, s, vh = np.linalg.svd(shiftedDataSample, full_matrices=False) # rows of vh contain the principal axes
        rotationMatrix = np.transpose(vh)
    rotatedData = shiftedData @ rotationMatrix
    xMin, yMin, zMin = np.amin(rotatedData, axis=0)
    xMax, yMax, zMax = np.amax(rotatedData, axis=0)
    if plot:
        SCALE = 3 # for visualization purposes only
        fig = plt.figure()
        ax = fig.add_subplot(121, projection="3d")
        ax.scatter(shiftedData[:, 0], shiftedData[:, 1], shiftedData[:, 2])
        ax.set_title("Original Data")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        for i in range(3):
            ax.plot([0, SCALE*vh[i, 0]], [0, SCALE*vh[i, 1]], [0, SCALE*vh[i, 2]]) # plot principal axes for visualization
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(rotatedData[:, 0], rotatedData[:, 1], rotatedData[:, 2])
        ax2.set_title("Rotated Centered Data")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        plt.show()
    if returnRotatedData:
        return rotationMatrix, xMin, yMin, zMin, xMax, yMax, zMax, rotatedData
    else:
        return rotationMatrix, xMin, yMin, zMin, xMax, yMax, zMax

def demo1():
    boundingBox(generateDummyData(45, 60, 120, 10, 3, 1, plot=False), plot=True)

# demo1()
