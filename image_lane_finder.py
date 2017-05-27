import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageLaneFinder:
    xm_per_pix = 3.7 / 700
    ym_per_pix = 30 / 720

    def __init__(self, camera, perspective_mtx, perspective_mtxInv, img_shape, preprocessor):
        self.camera = camera
        self.perspective_M = perspective_mtx
        self.perspective_Minv = perspective_mtxInv
        self.height, self.width = img_shape
        self.preprocessor = preprocessor

    def warp(self, img):
        return cv2.warpPerspective(img, self.perspective_M, (self.width, self.height), flags=cv2.INTER_LINEAR)

    def unwarp(self, image, left_fit, right_fit):
        ploty = np.linspace(0, self.height - 1, self.height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros((self.height, self.width)).astype(np.uint8)
        color_warp = np.asarray(np.dstack((warp_zero, warp_zero, warp_zero)), dtype=np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.perspective_Minv, (self.width, self.height))

        # calculate position
        left_pos = np.min(np.argwhere(newwarp[-1, :, 1]))
        right_pos = np.max(np.argwhere(newwarp[-1, :, 1]))
        position = (self.width / 2.0 - (left_pos + right_pos) / 2.0) * self.xm_per_pix
        side = "left" if position < 0 else "right"

        font = cv2.FONT_HERSHEY_SIMPLEX
        weighted = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        cv2.putText(weighted, "Position is {0:.2f} m {1} of center".format(abs(position), side), (400, 150), font, 1,
                    (255, 255, 255), 2)

        # calculate curvature
        curvature = self.measure_lanes_curvatures(left_fit, right_fit)
        text = "Radius of Curvature: {} m".format(int(curvature))
        cv2.putText(weighted, text, (400, 100), font, 1, (255, 255, 255), 2)

        return weighted

    def sliding_window_find_lane_indices(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[self.height * 0.5:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(self.height / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.height - (window + 1) * window_height
            win_y_high = self.height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        return left_lane_inds, right_lane_inds

    def fit_from_indices(self, binary, left_lane_inds, right_lane_inds):
        nonzeroy, nonzerox = self.nonzero_decompose(binary)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            return None, None

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def nonzero_decompose(self, binary):
        nonzero = binary.nonzero()
        return np.array(nonzero[0]), np.array(nonzero[1])

    def fit_from_previous_fit(self, binary, prev_left_fit, prev_right_fit):
        nonzeroy, nonzerox = self.nonzero_decompose(binary)
        margin = 100

        left_lane_inds = (
        (nonzerox > (prev_left_fit[0] * (nonzeroy ** 2) + prev_left_fit[1] * nonzeroy + prev_left_fit[2] - margin)) & (
        nonzerox < (prev_left_fit[0] * (nonzeroy ** 2) + prev_left_fit[1] * nonzeroy + prev_left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (
        prev_right_fit[0] * (nonzeroy ** 2) + prev_right_fit[1] * nonzeroy + prev_right_fit[2] - margin)) & (
                           nonzerox < (
                           prev_right_fit[0] * (nonzeroy ** 2) + prev_right_fit[1] * nonzeroy + prev_right_fit[
                               2] + margin)))

        return left_lane_inds, right_lane_inds

    def plot_poly_fit(self, binary, left_fit, right_fit, margin=100):
        nonzeroy, nonzerox = self.nonzero_decompose(binary)

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.height - 1, self.height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img = np.asarray(np.stack((binary, binary, binary), axis=2), dtype=np.uint8) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        # Plot the result
        f, ax1 = plt.subplots(1, 1, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(result)
        ax1.plot(left_fitx, ploty, color='yellow')
        ax1.plot(right_fitx, ploty, color='yellow')
        plt.imshow(result)

    def find_lanes(self, img, prev_left_fit=None, prev_right_fit=None):
        # step 1. Undistort
        undistorted = self.camera.undistort(img)

        # step 2. Warp
        warped = self.warp(undistorted)

        # step 3. Apply Gradient Threshold
        binary = self.preprocessor(warped)

        # step 4. Fit and locate lane lines
        if prev_left_fit == None or prev_right_fit == None:
            left_lane_inds, right_lane_inds = self.sliding_window_find_lane_indices(binary)
        else:
            left_lane_inds, right_lane_inds = self.fit_from_previous_fit(binary, prev_left_fit, prev_right_fit)

        left_fit, right_fit = self.fit_from_indices(binary, left_lane_inds, right_lane_inds)

        # step 5. undistort and draw the lanes
        return undistorted, left_fit, right_fit

    def draw_lanes(self, undistorted, left_fit, right_fit):
        return self.unwarp(undistorted, left_fit, right_fit)

    def measure_lanes_curvatures(self, left_fit, right_fit):
        left_curverad = self.measure_lane_curvature(left_fit)
        right_curverad = self.measure_lane_curvature(right_fit)
        return (left_curverad + right_curverad) / 2.0

    def measure_lane_curvature(self, fit):
        ploty = np.linspace(0, self.height - 1, self.height)
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * self.ym_per_pix, fitx * self.xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

        return curverad