import cv2
import numpy as np

class Camera:
    calibrated = False

    def __init__(self, calibration_images, chessboard_dims):
        self.calibrate(calibration_images, chessboard_dims)

    def calibrate(self, calibration_images, chessboard_dims):
        if len(calibration_images) == 0:
            return

        w, h = chessboard_dims

        # determine object points of the chessboard
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # assume all calibration images have the same shape
        img_shape = calibration_images[0].shape

        objpoints = []
        imgpoints = []
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_dims, None)
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[0:2], None, None)
        if ret:
            self.distortion_matrix = mtx
            self.distortion_coeffs = dist
            self.calibrated = True

    def undistort(self, img):
        if self.calibrated:
            return cv2.undistort(img, self.distortion_matrix, self.distortion_coeffs, None, self.distortion_matrix)
        return img  # if camera not calibrated, simply return the original image
