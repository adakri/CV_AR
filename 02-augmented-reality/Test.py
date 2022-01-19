#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 00:07:16 2022

@author: abdel_dakri
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


fname_template = "template.jpg"
template = cv.imread(fname_template)


fname_example = "example.jpg"
img = cv.imread(fname_example)
print("================done reading===============")

template = cv.cvtColor(template,cv.COLOR_BGR2GRAY)

#cv.imshow("template",template)
#cv.waitKey(500)

# do sift comparison between image and template=====================
# Start by finding best features in template
NUM_FEATURES = 50
sift = cv.SIFT_create(nfeatures = NUM_FEATURES)
keypoints, descriptors = sift.detectAndCompute(template,None)

#https://docs.opencv.org/3.4/de/d30/structcv_1_1DrawMatchesFlags.html


# Now for the example
keypoints_example, descriptors_example = sift.detectAndCompute(img,None)


# Feature matching
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

matches = bf.match(descriptors,descriptors_example)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(template, keypoints, img, keypoints_example, matches[:50], img, flags=2)
plt.imshow(img3),plt.show()

# Only select the matches
print(matches)



# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# I need to 2D it

objp = [ ( int(key.pt[0]), int(key.pt[1]), 0 ) for key in keypoints]
objp = np.array(objp, np.float32)
objpoints = np.array(objp, dtype=np.float32)


imgpoints = np.array([ ( int(key.pt[0]), int(key.pt[1]) ) for key in keypoints_example], dtype=np.float32)


print(imgpoints.shape)
print(objpoints.shape)

    
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera([objp], [imgpoints], template.shape[::-1], None, None)


print("===== The translation and rotation ========")
print(mtx)
print(rvecs)
print(tvecs)
print("===========================================")

h,  w = template.shape
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', img)


print("Number of Keypooints: ",len(objpoints))


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[0], tvecs[0], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2[0,0,:], cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )



# Find the rotation and translation vectors.
ret,rvecs, tvecs = cv.solvePnP(objpoints, imgpoints, mtx, dist)

print(rvecs)
print(tvecs)

# project 3D points to image plane
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)


def draw(img, corners, imgpts):
    imgpts = np.array(imgpts, dtype="float32")
    corner = tuple(corners[0].ravel())
    corner = (int(corner[0]), int(corner[1]))
    #img = cv.line(img, (0,0), (100,100), (255,0,0), 5)
    img = cv.line(img, corner, tuple(map(int, tuple(imgpts[0][0].ravel()))), (255,0,0), 5)
    img = cv.line(img, corner, tuple(map(int, tuple(imgpts[1][0].ravel()))), (0,255,0), 5)
    img = cv.line(img, corner, tuple(map(int, tuple(imgpts[2][0].ravel()))), (0,0,255), 5)
    return img


final = draw(img,imgpoints,imgpts)
imS = cv.resize(final, (640, 480))  
cv.imshow('img',imS)
k = cv.waitKey(8000)










cv.destroyAllWindows()