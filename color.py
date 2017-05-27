import cv2
import numpy as np

def hls_select(img, channel = 0, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    selected_channel = hls[:,:,channel]
    binary_output = np.zeros_like(selected_channel)
    binary_output[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
    return binary_output

def rgb_select(img, channel = 0, thresh=(0, 255)):
    selected_channel = img[:,:,channel]
    binary_output = np.zeros_like(selected_channel)
    binary_output[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
    return binary_output