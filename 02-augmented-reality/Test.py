#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 00:07:16 2022

@author: abdel_dakri
"""

import numpy as np
import cv2 as cv
import glob


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


fname = "calibration.jpg"

img = cv.imread(fname)

print("done reading")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

print(gray)

cv.imshow("gray",gray)

cv.waitKey(500)

ret, corners = cv.findChessboardCorners(gray, (9,6),None)

print(ret)

if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # Draw and display the corners
    cv.drawChessboardCorners(img, (7,6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(500)
        
    
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

h,  w = gray.shape
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )



# Find the rotation and translation vectors.
ret,rvecs, tvecs = cv.solvePnP(objpoints[0], corners2, mtx, dist)

print(rvecs)
print(tvecs)

# project 3D points to image plane
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

print(tuple(imgpts[0][0].ravel()))

print(tuple(corners[0].ravel()))

def draw(img, corners, imgpts):
    imgpts = np.array(imgpts, dtype="float32")
    corner = tuple(corners[0].ravel())
    corner = (int(corner[0]), int(corner[1]))
    #img = cv.line(img, (0,0), (100,100), (255,0,0), 5)
    img = cv.line(img, corner, tuple(map(int, tuple(imgpts[0][0].ravel()))), (255,0,0), 5)
    img = cv.line(img, corner, tuple(map(int, tuple(imgpts[1][0].ravel()))), (0,255,0), 5)
    img = cv.line(img, corner, tuple(map(int, tuple(imgpts[2][0].ravel()))), (0,0,255), 5)
    return img


gray = draw(img,corners2,imgpts)
cv.imshow('img',gray)
k = cv.waitKey(5000)










cv.destroyAllWindows()