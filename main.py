#!/usr/bin/python3
from multiprocessing import Pool
import numpy as np
import cv2
import math
import time
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d
import os
import itertools
from PIL import Image

import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.models import load_model
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

startTime = time.time()

# TODO: Decompose and clean up

def findPointDist(pointh, C):
    xDiff = pointh[0] - C[0]
    xTerm = math.pow(float(xDiff), 2)
    yDiff = pointh[1] - C[1]
    yTerm = math.pow(float(yDiff), 2)
    zDiff = pointh[2] - C[2]
    zTerm = math.pow(float(zDiff), 2)
    distance = math.sqrt(xTerm + yTerm + zTerm)
    return abs(distance)


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 2e-6


# TRANSFORM COORDINATES:
# ----------------------------------
# Given camera position, pixel data, and point cloud data, creates (U,V) coordinate transformation that assigns each pixel in the
# image to a 3D point while ignoring points not in the image's field of view.
# Parameters: 3D point cloud, camera position matrix, transformation matrix, pixel matrix, camera coordinate

# Returns: Transformed (UV) coordinates, image projected on point cloud
def transformCoords(points3D, camMat, rtMat, RGB3D, C):
    start = time.time()
    uv = np.zeros((len(points3D), 2))
    projImg = np.full((2448, 2448, 4), 255) #Initialize a white 2448 * 2448 photo with an additional layer. That 4th layer will be used to assign index of 3D point that goes into that pixel.
    projImg[:, :, 3] = 0
    i = 0
    for point in points3D:
        pointh = np.append(point, 1)  #Generate [X,Y,Z,1] vector. 1 added for math purposes.
        sh = np.dot(camMat, np.dot(rtMat, pointh))  #Perform operation in the first equation at: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        s = sh / sh[2]  #Above continued. 3D points [X,Y,Z] are projected onto 2D plane [s[0],s[1]]
        uv[i, 0] = s[0] #Convert s into uv for 2D visualization purposes only
        uv[i, 1] = s[1] #Convert s into uv for 2D visualization purposes only
        if s[0] >= 0 and s[0] <= 2447 and s[1] >= 0 and s[1] <= 2447:  #Only consider the points inside the field of view of camera (2448 x 2448)
            pixelX = int(np.rint(s[0])) #Round s values to the closest integer to place them on 2448x2448 grid.
            pixelY = int(np.rint(s[1])) #Round s values to the closest integer to place them on 2448x2448 grid.
            pointDist = findPointDist(pointh, C) #Calculate the distance between point and camera to be used for checking which points are in front.
        i += 1
    end = time.time() - start
    projImg = np.swapaxes(projImg, 0, 1)
    #print(str(end) + ' seconds for transformCoords function.')
    return uv, np.flip(projImg, axis=0), pointCamDistances
    # TODO: store pointDist for each image. Maybe in a (n,m) array where n = # of total points, m = # of photos.
    # (cont) add another layer to projImg: projImg = np.full((2448, 2448, 5), 255)
    # (cont) In last layer of projImg store which image is closest (eg:0,1,2,...)

# PREDICT 2D:
# -----------------------------------
# Uses trained machine learning data to predict whether or not a crack exists in the given pixel matrix.
# Parameters: Image, size of image segment, model classifier to be used
# Returns: Image with segments either detected(1) or not(0)
def predict2d(img, input_size, model):
    start = time.time()
    H, W, C = img.shape
    h, w = input_size
    results = np.zeros((int(H // h), int(W // w)))
    for i in range(int(H // h)):
        for j in range(int(W // w)):
            prediction = model.predict(sub_img, batch_size=1)
            if prediction[0][1] > 0.5:
                results[i, j] = 1
    end = time.time() - start
    # print(str(end) + ' seconds for predict2d function.')
    return results


# RUN CONVNET:
# ------------------------------------
# Uses keras model to detect cracks in 2D image.
# Parameters: High resolution 2D image
# Returns: Same 2D image with cracks highlighted in red
def runConvnet(image, rundir, nnImgSize):
    start = time.time()
    classifier = load_model(
        rundir + 'FractchaDatConvNetModel_CS231N_v1_wMETU_ZHANG.h5')  # Put the FratchaDat Model in the rundir
    img = Image.open(image)
    img = np.asarray(img) + 1.0 - 1.0
    cracked_img_predictions = predict2d(img[:, :, :3], (nnImgSize, nnImgSize), classifier) # predict2d is giving us a 2D matrix of size (2448/32,2448,32). Basically subdividing the photo into 32x32 blocks and labeling each block as cracked or uncracked.
    for i in range(cracked_img_predictions.shape[0]): #First march along horizontal
        for j in range(cracked_img_predictions.shape[1]): #Then march along vertical
            if cracked_img_predictions[i, j] == 1: # If the 32x32 block under consideration is predicted as cracked:
                img[i * nnImgSize:i * nnImgSize + nnImgSize, j * nnImgSize:j * nnImgSize + nnImgSize, 0] = 255 #Change the color of that portion in the high-res image into red.
    img = img / 255
    img = np.flip(img, axis=0)
    # plt.figure()
    # plt.imshow(img)
    # plt.imsave(str(image) +"_highlighted.jpg", np.flip(img, axis=0))
    end = time.time() - start
    # print(str(end) + ' seconds for runConvnet function.')
    return img, np.flip(cracked_img_predictions, axis=0) #Output modified image and convnet predictions matrix (flipped)


# SET CAMERA MATRIX:
# --------------------------------------
# Initializes the camera matrix with focal lengths and principal image points.
# Returns: Camera matrix for Samsung Galaxy Tab S2 (w/ these current numbers! Change if camera is changed)
def setCamMat():
    cameraMatrix = np.zeros((3, 3))
    cameraMatrix[0][0] = 2.61267502e+03   # Focal length in X (in pixels). p = (2.91 / 3) * 2448 = 2375
    cameraMatrix[1][1] = 2.61354582e+03  # Focal length in Y (in pixels). p = (2.91 / 3) * 2448 = 2375
    cameraMatrix[2][2] = 1
    cameraMatrix[0][2] = 1.20193956e+03  # Principal point in x (usually image center = 2448/2)
    cameraMatrix[1][2] = 1.21307617e+03  # Principal point in y (usually image center = 2448/2)
    # [[2.61267502e+03 0.00000000e+00 1.20193956e+03]
    #  [0.00000000e+00 2.61354582e+03 1.21307617e+03]
    # [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    return cameraMatrix


# READ POINT CLOUD:
# ----------------------------------------
# Reads a given point cloud file and returns different parts of its data.
# Parameters: Point cloud file
# Returns: Points from file (N,3), Colors from file (N,3), and Integer(0-255) colors from file
def readPointCloud(rundir, ptCloudFile):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pcd = o3d.io.read_point_cloud(rundir + ptCloudFile)
    return pcd, np.asarray(pcd.points), np.asarray(pcd.colors), np.asarray(pcd.colors) * 255


# GET ROTATION/TRANSLATION MATRICES
# -----------------------------------------------
# Takes a single line from a .csv or .txt file and creates a rotation matrix and translation vector from its data.
# Parameters: String from file line
# Returns: Rotation/Translation matrix
def getRTMat(line):
    x_rotation_adjustment = 0
    y_rotation_adjustment = 0
    z_rotation_adjustment = 0
    rData = []
    tData = []
    imageFile = line.split(",")[0]  # Unused for now
    data = ((line.split(",")[1]).split(" "))[:12]  # Trims line to relevant data
    data = [float(element) for element in data]
    for element in data:
        if (data.index(element) + 1) % 4 == 0:
            tData.append(element)
        else:
            rData.append(element)
    tData = [element * -1 for element in tData]  # Corrects the negative DotProduct bug
    rMat = np.array(rData).reshape(3, 3)
    tMat = np.array(tData).reshape(3, 1)
    rMat *= -1
    tMat[0][0] += -62  # X distance between cameras -62
    tMat[1][0] += 5    # Y distance between cameras 5
    tMat[2][0] += -25  # distance between cameras -25
    rtMat = np.concatenate((rMat, tMat), axis=1)
    rt44 = np.append(rtMat, np.array([[0, 0, 0, 1]], dtype='float'), axis=0)
    RcC = np.linalg.inv(rt44)
    C = RcC[0:3, 3]
    return rtMat, C, imageFile


rundir = 'C:/Users/DigitalTwins/Desktop/Demo_Run_Directory/'  # Michael Directory
ptCloudFile = None
csvFile = None
for file in os.listdir(rundir):
    if file[-4:] == '.ply':
        ptCloudFile = file
    if file[-4:] == '.csv' or file[-4:] == '.txt':
        csvFile = file
cameraMatrix = setCamMat()
nnImgSize = 32
pcd, points3D, RGB3D, RGB3Duint8 = readPointCloud(rundir, ptCloudFile)
file = open(rundir + csvFile)
lines = file.readlines()
for line in lines:
    if line == '\n':
        lines.remove(line)
line_tuples = []
lineno = 0
for line in lines:
    myTuple = [str(line), lineno]
    line_tuples.append(myTuple)
    lineno += 1


def parse(mylist):
    global points3D
    global cameraMatrix
    global RGB3Duint8
    global nnImgSize
    global RGB3D
    global pcd
    global lines
    rtMat, C, imageFile = getRTMat(mylist[0])
    uv2d, projImg, pointCamDists = transformCoords(points3D, cameraMatrix, rtMat, RGB3Duint8, C)

    # if line.split(",")[0] == "2019_07_19__15_37_56_HiResShot_0000.jpg":
    #     plt.figure()
    #     plt.imshow(projImg)
    #     plt.show()
    img, crackedMatrix = runConvnet((rundir + imageFile), rundir, nnImgSize)
    
    # Highlight the cracks in 3D point cloud based on 2D Convnet prediction.
    suspects = list()
    boxPerDim = crackedMatrix.shape[0] #2448/32
    for ver in range(boxPerDim): #First march along vertical
        for hor in range(boxPerDim): #Then march along horizontal
            #TODO: Check which image is closest to the current 32x32 block (already in the 5th layer)
            #(cont) cameraCounts = np.bincount(projImg[ver*nnImgSize : (ver+1)*nnImgSize, hor*nnImgSize : (hor+1)*nnImgSize,4])
            #(cont) closestCam = np.argmax(cameraCounts)
            if crackedMatrix[ver, hor] == 1 #TODO:and closestcam == image_pair[1](??): # If the 32x32 block under consideration is predicted as cracked and (TODO: If the high-res photo we're at is the closest to that box:
                projImg[ver*nnImgSize : (ver+1)*nnImgSize, hor*nnImgSize : (hor+1)*nnImgSize, 0:3] = np.array([255, 0, 0]) # Paint the 2D projection 3D points to the red
                suspects.append(projImg[ver * nnImgSize:(ver + 1) * nnImgSize, hor * nnImgSize:(hor + 1) * nnImgSize, 3]) #Append the IDs of 3D points to the list where we store the cracked 3D points.
    nonzerosuspects = list(filter(lambda a: a != 0, flatsuspects)) #Only leave the point IDs that are predicted as cracked.
    RGB3D[nonzerosuspects, :] = [1, 0, 0] 
    pcd.colors = o3d.utility.Vector3dVector(RGB3D) #Change the color of the point cloud.
    # Visualize modified 2D image
    # plt.imshow(projImg[:,:,0:3], origin='lower')
    return projImg, img


if __name__ == '__main__':
    p = Pool(len(lines))
    image_list = p.map(parse, line_tuples)
    subplot_index = 1
    plt.figure()
    for i in range(0, 2):
        for image_pair in image_list:
            plt.subplot(2, len(image_list), subplot_index)
            plt.grid()
            if i == 1:
                plt.imshow(image_pair[i], origin='lower')
            else:
                plt.imshow((image_pair[i]).astype(np.uint8), origin='lower')
            subplot_index += 1
    # Save modified 3D point cloud
    # o3d.io.write_point_cloud(rundir+ptcloudfile[:-4]+"_highlighted.ply",pcd)
    endTime = time.time() - startTime
    plt.show()
    # for image in image_list:
    #     for i in range(0,2):
    #         plt.subplot(1, 2, subplot_index)
    #         if i == 0:
    #             img = mpimg.imread(rundir + image[i])
    #             imgplot = plt.imshow(img)
    #         else:
    #             plt.imshow(image[i])
    #         subplot_index += 1
    #     plt.show()
    #     subplot_index = 1
    print(str(int(endTime)) + ' seconds for entire program.')