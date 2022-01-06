#!/usr/bin/env python3
""" computer vision AR demo"""

# ------- imports ------------------------------------------------------------
import sys
import logging
import argparse as ap
import cv2
import numpy as np


# ------- argument parser ----------------------------------------------------
parser = ap.ArgumentParser(description='Simple augmented reality application.')
parser.add_argument('file', metavar='video_file', type=str, nargs='?',
                    help='a video file to apply AR on')
parser.add_argument('-o', '--output', type=str, nargs='?',
                    help='save live input stream as a video file')
parser.add_argument('-g', '--grid-size', type=int, nargs=2, default=(6, 5),
                    help='calibration grid size')
parser.add_argument('-l', '--log', choices=['NOTSET', 'DEBUG', 'INFO',
                                            'WARNING', 'ERROR', 'CRITICAL'],
                    help='Log level used to display logs', default='INFO')

args = parser.parse_args()

# logging
fmt = '[%(asctime)s.%(msecs)-3.2s%(levelname)s] %(message)s'
logging.basicConfig(level=getattr(logging, args.log),
                    format=fmt, datefmt='%H:%M:%S')
logging.info('Python version ' + sys.version)
logging.info('Arguments: ' + str(args))


# ------- draw augmentation object (cube) -------------------------------------
# function to draw our augmented scene / cube
def draw(image, points2d):
    # TODO 3: draw virtual object on real frame
    return image


# ------- main initialization and loop ----------------------------------------
def main():

    # get source video capture object
    cam = cv2.VideoCapture(0 if args.file is None else args.file)

    # define the codec & create VideoWriter object in case we write video output
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        dim = int(cam.get(3)), int(cam.get(4))  # float width and height
        out = cv2.VideoWriter(args.output, fourcc, 20.0, dim)

    # open window where we will show video stream
    cv2.namedWindow("Calibration and augmented reality demo")
    
    # construct array of checkerboard coordinates in object coordinate frame
    grid = args.grid_size
    object_points = [[i, j, 0] for j in range(grid[1]) for i in range(grid[0])]
    object_points = np.array(object_points, np.float32)
    print(f'{object_points=}')

    # Arrays to store object points and image points from all the images.
    points3d = []    # 3d points of calibration in its coordinate frame
    points2d = []    # matching 2d points in image plane.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*6,3), np.float32)
    
    objp[:,:2] = np.mgrid[0:5,0:6].T.reshape(-1,2)
    
    #print(objp)
    
    # variables that will contain calibration result
    intrinsics, distortion = None, None

    # initialize coordinates that will be used to draw the cube
    cube = np.float32([[0, 0, 0],  [0, 3, 0],  [3, 3, 0],  [3, 0, 0],
                       [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    print('\n\nWave the calibration checkboard to the camera for a few frames, '
          'then press C to calibrate and see AR cube on checkerboard.\n')
    print('Press ESC to close.')

    # corner refinement termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_counter = 0
    scale = 0.3

    # main loop ---------------------------------------------------------------
    while cam.isOpened():
        # read an image and get return code
        ret, image = cam.read()

        # exit loop if we reach the end of stream
        if not ret:
            break
        
        # barrier of image existence

        # read key
        k = cv2.waitKey(1)

        # gray image conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # examine key events
        if k & 0xFF == 27:
            # exit if ESC pressed
            logging.info("Escape hit, closing...")
            break
        elif k & 0xFF == ord('c') and intrinsics is None:
            # when user presses 'c', perform calibration
            # TODO 2.1: use the point3d <-> point2d matches to calibrate
            print("==Calibrating==")
            _, intrinsics, distortion, _, _ = cv2.calibrateCamera(points3d, points2d, gray.shape[::-1], None, None)

            logging.info('Calibration done')
            logging.info(f'Estimated distortion\n{distortion}')
            logging.info(f'Estimated intrinsics matrix\n{intrinsics}')
            
            
           
            

        if out:
            out.write(image)

        # TODO 1.1: chessboard detection on scaled down image for improved speed
        ret, corners = cv2.findChessboardCorners(gray, (5,6), None)
       
        if ret:
            print("============ corners in a frame had been detected =================")
             
            # TODO 1.3: if corners found, get sub-pixel refinement of corners
            #print(corners)
            
            search_window_size = 7 # this will be multiplied by 2
            dead_zone_size = -2 
            
            corners = cv2.cornerSubPix(gray,corners, (search_window_size,search_window_size), (dead_zone_size,dead_zone_size), criteria)
            
            
            image_counter += 1
            if intrinsics is None:    # calibration ongoing

                # only take one in 30 frames for calibration to avoid overload
                if image_counter % 30 == 0:
                    # TODO 2.1: append each new point2d detection, and append
                    # the corresponding object point coordinates each time
                    points3d.append(objp)
            
                    points2d.append(corners)
                    

                    # show frames used for calibration as inverted
                    image = cv2.bitwise_not(image)

                # TODO 1.2: draw and display the corners                
                #image.append(corners)
                                
                cv2.drawChessboardCorners(image, (5,6), corners, ret)
            
            else:
                # now that camera is calibrated, we can estimate extrinsics
                # TODO 2.2  find rotation and translation vectors.
                
                points3d = np.array(points3d, dtype='float32')
                points2d = np.array(points2d, dtype='float32')
                
                """
                print("points3d shape")
                print(points3d.shape)
                print(len(points3d.shape))
                """
                
                
                len_obj = 0
                
                if(len(points3d.shape) == 2):
                    
                    m, _ = points3d.shape
                    points3d = points3d.reshape(m,3)
                    points2d = points2d.reshape(m, 2)
                    len_obj = m
                else:
                    m, n, _ = points3d.shape
                    points3d = points3d.reshape(n * m,3)
                    points2d = points2d.reshape(n * m, 2)
                    len_obj = n*m
                
                print("Shapes of point3d/2d")
                print(points3d.shape)
                print(points2d.shape)
                
                print("intrinsics shape")
                print(intrinsics.shape)
                print(type(intrinsics))
                
                ret, rotation_vectors, translation_vectors, _ = cv2.solvePnPRansac(points3d, points2d, intrinsics, distortion)
                
                """
                print("====Rot and trans vectors====")
                print(rotation_vectors)
                
                print(translation_vectors)
                """
                
                # TODO 2.3 project 3D points to image using estimated parameters
                
                #projected_points, jacobian = cv2.projectPoints(points3d, translation_vectors, rotation_vectors,  intrinsics, distortion)
                
                #print(projected_points.shape)
                projected_point, jacobian = cv2.projectPoints(points3d, translation_vectors, rotation_vectors,  intrinsics, distortion)
                
                def compute_projection_error(projected_point):
                    error = 0.
                    print("===##=== inside the error function ===##===")
                    print(points2d.shape)
                    for i in range(len_obj):
                        #print("the error")                        
                        error += np.sqrt(np.linalg.norm(projected_point[i] - points2d[i]))
                        #error += cv2.norm(projected_point, points2d[i], cv2.NORM_L2)    
                    print( "total error: {}".format(error/len_obj) )
                 
                compute_projection_error(projected_point)
                
                
                #image = draw(image)
                
                cv2.waitKey(1000)
                
                def draw(img, corners, imgpts):
                    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
                    print("======== drawing ========")
                    corner = tuple(corners[17].ravel())
                    
                    # project 3D points to image plane
                    imgpts, jac = cv2.projectPoints(axis, translation_vectors, rotation_vectors,  intrinsics, distortion)
                    
                    print("====Rot and trans vectors====")
                    print(rotation_vectors)
                    print(translation_vectors)
                    
                    
                    
                    print(imgpts.shape)
                    print(corner)
                    print(tuple(map(int, tuple(imgpts[1].ravel()))))
                    #img = draw(img,corners2,imgpts)
                    
                    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, tuple(imgpts[0].ravel()))), (255,0,0), 5)
                    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, tuple(imgpts[1].ravel()))), (0,255,0), 5)
                    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, tuple(imgpts[2].ravel()))), (0,0,255), 5)
                    return img
                
                
                def draw_cube(img, corners, imgpts):
                    print("======== drawing cube ========")
                    # project 3D points to image plan
                    imgpts, jac = cv2.projectPoints(cube, translation_vectors, rotation_vectors,  intrinsics, distortion)
                    # draw ground floor in green
                    imgpts = np.array(imgpts, dtype="float32")
                    
                    print(type(imgpts))
                    
                    img = cv2.drawContours(img, imgpts[:4].astype(int),-1,(0,255,0),-3)
                    # draw pillars in blue color
                    for i,j in zip(range(4),range(4,8)):
                        img = cv2.line(img, tuple(map(int, tuple(imgpts[i].ravel()))), tuple(map(int, tuple(imgpts[j].ravel()))),(0,0,255),3)
                    # draw top layer in red color
                    img = cv2.drawContours(img, imgpts[4:].astype(int),-1,(0,0,255),3)
                    return img
                
                image = draw(image, corners, projected_point)
        
        
        
        
        cv2.imshow('img', image)

    # finalize and release stream objects, input and output when applicable
    cam.release()
    if out:
        out.release()
        
        
        
if __name__ == "__main__":
    main()

  