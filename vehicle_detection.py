"""
Vehicle detection routines +  video procesing pipeline

Created on 11.08.2017

@author: harald
"""
from skimage.feature import hog
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

####################
# Swap color planes of the image
def cv2mpimg(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


####################
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True):
    res = hog(img,
              orient,
              (pix_per_cell, pix_per_cell),
              (cell_per_block, cell_per_block),
              block_norm = 'L1',
              visualise = vis,
              feature_vector = feature_vec)
    return res


####################
# Extract features
def extract_features(img, orientation, pix_per_cell, cell_per_block, toVector=True, visualize=False):
    res = get_hog_features(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:,:,0], 
                           orientation, 
                           pix_per_cell, 
                           cell_per_block, 
                           vis=visualize, 
                           feature_vec=toVector)
    if visualize:
        return res[0], res[1]
    else:
        return res, None
    

####################
if __name__ == '__main__':
    pass
