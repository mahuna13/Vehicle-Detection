from image_lane_finder import *

# main code
class VideoLaneFinder:
    def __init__(self, camera, perspective_mtx, perspective_mtxInv, img_shape, preprocessor):
        self.init_values()
        self.frame_lane_finder = ImageLaneFinder(camera, perspective_mtx, perspective_mtxInv, img_shape, preprocessor)

    def init_values(self):
        self.prev_left = None
        self.prev_right = None

    def process_frame(self, image):
        undistorted, left_fit, right_fit = self.frame_lane_finder.find_lanes(image, self.prev_left, self.prev_right)

        if self.prev_left == None:
            # first fit found
            self.prev_left = left_fit
            self.prev_right = right_fit
        else:
            if left_fit != None:
                potential_left_fit = left_fit * 0.2 + self.prev_left * 0.8
                left_diff = abs(potential_left_fit - self.prev_left)
                if abs(self.frame_lane_finder.measure_lane_curvature(potential_left_fit) < 10000):
                    if not (left_diff[0] > 0.001 or left_diff[1] > 1.0 or left_diff[2] > 100):
                        self.prev_left = potential_left_fit
            if right_fit != None:
                potential_right_fit = right_fit * 0.2 + self.prev_right * 0.8
                right_diff = abs(potential_right_fit - self.prev_right)
                if abs(self.frame_lane_finder.measure_lane_curvature(potential_right_fit) < 10000):
                    if not (right_diff[0] > 0.001 or right_diff[1] > 1.0 or right_diff[2] > 100):
                        self.prev_right = potential_right_fit

        return self.frame_lane_finder.draw_lanes(undistorted, self.prev_left, self.prev_right)

    def find_lanes(self, video):
        self.init_values()
        output_video = video.fl_image(self.process_frame)
        return output_video