"""
Written by Ruiqi Chen and Grayson Armour
8 August 2019
This code takes in a .ply point cloud and finds a minimum bounding box using SVD.
It then centers and rotates the original point cloud to be aligned with Cartesian X, Y, and Z axes.
For efficiency, by default, point sampling is enabled. This can be disabled with the appropriate input arguments.
Saves the bounded and centered point cloud as a .pts file in the run directory.
"""

import numpy as np
import numpy.matlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def PLYtoPTS(rundir):
    # Finds the .ply point cloud file and resaves it as a manipulatable .pts file. Then reads that file's lines into
    # strings for editing.
    ptCloudFile = None
    for file in os.listdir(rundir):
        if file[-4:] == '.ply':
            ptCloudFile = file
    pcd = o3d.io.read_point_cloud(rundir + ptCloudFile)
    o3d.write_point_cloud('temp.pts', pcd) # Rewrite to temporary .pts

    file = open(rundir + 'temp.pts')
    lines = file.readlines()
    for line in lines:   # Remove empty lines
        if line == '\n':
            lines.remove(line)
    file.close()
    os.remove(rundir + 'temp.pts') # delete temporary .pts file
    return lines, ptCloudFile

def parseLines(lines):
    # Returns data parsed from line of .pts file, including XYZ, IRGB, and header data.
    header = lines.pop(0)
    n = len(lines)
    xyz = np.zeros((n, 3), dtype=float)
    irgb = np.zeros((n, 4), dtype=int) # IRGB data must be ints
    row = 0
    for line in lines:
        lineData = np.fromstring(line, dtype=float, sep=' ')
        xyz[row][0] = lineData[0]
        xyz[row][1] = lineData[1]
        xyz[row][2] = lineData[2]
        irgb[row][0] = lineData[3]
        irgb[row][1] = lineData[4]
        irgb[row][2] = lineData[5]
        irgb[row][3] = lineData[6]
        row += 1
    return header, xyz, irgb

def findPointCentroid(data):
    # data is an n x 3 or 3 x n numpy matrix containing all x, y, z coordinate of points. Finds centroid relative to
    # points and point density
    if data.shape[0] == 3:
        return np.mean(data, axis=1)
    elif data.shape[1] == 3:
        return np.mean(data, axis=0)
    else:
        raise Exception("Input data must be of dimensions n x 3 or 3 x n!")

def findBoxCentroid(data):
    # data is an n x 3 or 3 x n numpy matrix containing all x, y, z coordinate of points. Finds centroid of bounding box.
    xMin, yMin, zMin = np.amin(data, axis=0)
    xMax, yMax, zMax = np.amax(data, axis=0)
    centroid = np.zeros(3)
    x = (xMax + xMin) / 2
    y = (yMax + yMin) / 2
    z = (zMax + zMin) / 2
    centroid[0] = x
    centroid[1] = y
    centroid[2] = z
    return centroid

def boundingBox(data, returnRotatedData=True, sampleSize=None):
    # data is an n x 3 or 3 x n numpy matrix containing all x, y, z coordinate of points
    # for efficiency, sampling is enabled by default. To disable sampling, change sampleSize to None.
    # if data contains fewer points than the samplingSize, no sampling will take place.
    if data.shape[0] == 3:
        data = np.transpose(data)
    centroid = findPointCentroid(data) # centroid for rotation
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
    centroid = findBoxCentroid(rotatedData)  # move centroid to center of bounded box
    rotatedCenteredXYZ = rotatedData - numpy.matlib.repmat(centroid, rotatedData.shape[0], 1)
    if returnRotatedData:
        return rotatedCenteredXYZ
        # return rotationMatrix, xMin, yMin, zMin, xMax, yMax, zMax, rotatedData
    else:
        return rotationMatrix, xMin, yMin, zMin, xMax, yMax, zMax

def writeToPTS(rundir, header, xyz, irgb):
    # Write new points and their header into a new .pts file
    newFile = open(rundir + 'bounded.pts', 'w+')
    row = 0
    newFile.write(str(header))
    rowNum = 0
    for row in xyz:
        x = str(row[0])
        y = str(row[1])
        z = str(row[2])
        i = str(irgb[rowNum][0])
        r = str(irgb[rowNum][1])
        g = str(irgb[rowNum][2])
        b = str(irgb[rowNum][3])
        newFile.write(x + ' ' + y + ' ' + z + ' ' + i + ' ' + r + ' ' + g + ' ' + b + '\n')
        rowNum += 1
    newFile.close()

if __name__ == '__main__':
    # rundir is filepath to location containing .ply point cloud file
    rundir = '/Users/graysonarmour/OneDrive - Leland Stanford Junior University/Digital Twin/boundingbox/'
    lines, ptCloudFile = PLYtoPTS(rundir)
    header, xyz, irgb = parseLines(lines)
    rotatedCenteredXYZ = boundingBox(xyz) # Rotate and center points about centroid
    writeToPTS(rundir, header, rotatedCenteredXYZ, irgb) # Write into a readable format

    # VISUALIZE IN OPEN3D
    pcd = o3d.read_point_cloud(rundir + 'bounded.pts')
    mesh_frame = o3d.geometry.create_mesh_coordinate_frame(size=150,
                                                           origin=[0, 0, 0])  # Show point cloud and coordinate frame
    o3d.visualization.draw_geometries([mesh_frame, pcd])

