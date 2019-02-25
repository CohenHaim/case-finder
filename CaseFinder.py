'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#                              Case Finder                                #
# A script that detectes game cases images and prints the game's console  #
# 
#
#
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


def isLogoExist(image_path, logo_path, visual=False, accuracy=0.8):
    # args: case image path, logo image path, visualize, matching accuracy (from 0-1)
    # returns: boolean - if found a match between game case and logo

    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE) # queryiamge
    
    # Features
    sift = cv2.xfeatures2d.SIFT_create()
    kp_logo, desc_logo = sift.detectAndCompute(logo, None)
    
    # Feature Matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # trainimage
    image = cv2.imread(image_path ,cv2.IMREAD_COLOR)
    image_w, image_h, _ = image.shape

    # Resizing if image is too small
    resize_at = 600
    resize_ratio = 2.5
    if image_w < resize_at and image_h < resize_at:
        image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio)

    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Finding and storing good points
    kp_grayimage, desc_grayimage = sift.detectAndCompute(grayimage, None)
    matches = flann.knnMatch(desc_logo, desc_grayimage, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)

    # Homography
    if len(good_points) > int(12*accuracy):
        query_pts = np.float32([kp_logo[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayimage[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective Transform
        h, w = logo.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        if not (matrix is None) and visual:
            dst = cv2.perspectiveTransform(pts, matrix) 
            homography = cv2.polylines(image, [np.int32(dst)], True, (255, 0, 255), 3) 
            cv2.imshow("Homography", homography)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return True

    return False

#folders paths and images extraction
logo_path = "logos/"
cover_path = "covers/"
logo_files = [f for f in listdir(logo_path) if isfile(join(logo_path, f))]
cover_files = [f for f in listdir(cover_path) if isfile(join(cover_path, f))]

foundMatch = 0
console_types = ["playstation", "xbox", "pc"]

#compere all covers with logs
if cover_files and logo_files:
    for cover_name in cover_files:
        for logo_name in logo_files:           
            cn = cover_path + cover_name
            ln = logo_path + logo_name

            if isLogoExist(cn, ln):
                #printing and matching the correct console name
                console_type = [t for t in console_types if logo_name[1] == t[1]][0]
                print(cover_name, "| Console Type:", console_type)
                foundMatch+=1
                break
        else:
            print(cover_name, " no match found")
else:
    print("Error - Missing files in covers/logos folders")
print("Found:", foundMatch, "/", len(cover_files))
