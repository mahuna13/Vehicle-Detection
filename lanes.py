import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os
import math
import glob
from collections import deque

# local packages
from gradient import *
from camera import *
from color import *


def thresholding_pipeline(img, s_thresh=(200, 255), r_thresh=(230, 255), sx_thresh=(70, 100)):
    img_cp = np.copy(img)
    r_binary = rgb_select(img_cp, 0, r_thresh)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img_cp, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sxbinary = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=7, thresh=(30, 90))
    lxbinary = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=7, thresh=sx_thresh)

    # Threshold color channel
    s_binary = hls_select(img_cp, 2, s_thresh)

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((lxbinary, sxbinary, s_binary))

    # Combine the two binary thresholds
    gray = cv2.cvtColor(img_cp, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(20, 105))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(20, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=3, thresh=(0.0, 0.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return (r_binary == 1) | (sxbinary == 1)  # (dir_binary == 1) & (gradx == 1) #| (r_binary == 1) | (sxbinary == 1)
