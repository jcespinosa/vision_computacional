#!/usr/bin/python

########################################################################
# This program is free software: you can redistribute it and/or modify #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or    #
# at your option) any later version.                                   #
#                                                                      #
# This program is distributed in the hope that it will be useful,      #
# but WITHOUT ANY WARRANTY; without even the implied warranty of       #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        #
# GNU General Public License for more details.                         #
#                                                                      #
# You should have received a copy of the GNU General Public License    # 
# along with this program.  If not, see <http://www.gnu.org/licenses/>.#
########################################################################

import cv2 as cv
import numpy as np
from sys import argv

# ==========================================================================
# surfDetector
#
# Gets an external filename
# Uses the SURF Feature detection method to obtain the keypoints
# and descriptors from the image
# Saves the keypoints and descriptors on a *.npy file
# BUG: Cannot save the keypoints
# Show the features in a cv windo
#
# ==========================================================================

# Paths to save the corresponding data,please create the folders before use the tool
keypointsPath = "./keypoints/"
descriptorsPath = "./descriptors/"
arrayPath = "./images/"
imagePath = "./signals/"

image = imagePath + argv[1]
name = argv[1].split(".")[0]

im2 = cv.imread(image)
im = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
surfDetector = cv.FeatureDetector_create("SURF")
surfDescriptorExtractor = cv.DescriptorExtractor_create("SURF")
keypoints = surfDetector.detect(im)
(keypoints, descriptors) = surfDescriptorExtractor.compute(im,keypoints)

np.save(descriptorsPath + name + ".npy", descriptors)
np.save(arrayPath + name + ".npy", im)
#with open(keypointsPath + nombre + ".kpt","w") as outputFile:
#    outputFile.write(",".join(keypoints))


for kp in keypoints:
    x = int(kp.pt[0])
    y = int(kp.pt[1])
    cv.circle(im, (x, y), 3, (0, 0, 255))

while True:
    cv.imshow("features", im)

    if cv.waitKey(10) == 10:
        break
