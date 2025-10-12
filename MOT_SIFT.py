# Original code from https://github.com/abhishek30-ml/Multiple-Object-Tracking/blob/main/mot.py
# Also, referenced in our project scope
# Updated and refactored into a class structure.

import numpy as np

# Helper functions are also from https://github.com/abhishek30-ml/Multiple-Object-Tracking/blob/main/mot.py
from MOT_SIFT_SCRIPTS import initialise, matching, tracker, result
from MOT_SIFT_SCRIPTS.sift_descriptor import Sift
from MOT_SIFT_SCRIPTS.kalman_filter import KalmanFilter


class MOT_SIFT:
    """
    Refactored MOT Tracker to process video frame by frame.
    """

    def __init__(self, detection_conf=0.4, sift_good_dist=300.0,
                 min_sift_score=20.0, accumulate_sift=3, max_age=30):
        self.model = None
        self.kf = KalmanFilter()
        self.sift = Sift(sift_good_dist)
        self.detection_conf = detection_conf
        self.min_sift_score = min_sift_score
        self.accumulate_sift = accumulate_sift
        self.max_age = max_age

        self.active_tracks = []
        self.offline_tracks = []
        self.next_id = 1

    def process_frame(self, frame, frame_no):
        """
        Processes a single video frame and returns the tracking results.

        Args:
            frame (np.ndarray): The input video frame.
            frame_no (int): The current frame number.

        Returns:
            np.ndarray: A NumPy array of tracks in the format [x1, y1, x2, y2, track_id, conf, class_id].
        """
        # Detection
        results = self.model(frame, verbose=False, conf=self.detection_conf)
        mesur_list = initialise.collect_measurement(results)
        des_list = self.sift.collect_descriptors(mesur_list, frame)
        # Matching
        unmatches = mesur_list.copy()
        if self.active_tracks:
            C, B = matching.maha_dist_matrix(mesur_list, self.active_tracks, self.kf)
            C2, B2 = matching.sift_dist_matrix(des_list, self.active_tracks, self.sift, self.min_sift_score,
                                               self.accumulate_sift)

            # The function now works correctly, so we can trust its return values.
            # The 'unmatches' variable will now be correctly updated.
            self.active_tracks, unmatches, des_list = matching.matching_assignment(
                C, B, C2, B2, self.active_tracks, unmatches, des_list, frame_no, self.kf
            )
        # Track Initialisation and Update
        unmatches_track, self.next_id = initialise.new_track(unmatches, des_list, self.next_id, frame_no, self.kf)

        self.active_tracks, self.offline_tracks = tracker.update_track(self.active_tracks, unmatches_track,
                                                                       self.offline_tracks, self.kf, self.max_age)

        # Format the output to match YOLO's .track() method
        # The required format is a NumPy array of [x1, y1, x2, y2, track_id, conf, class_id]
        formatted_tracks = []
        for track in self.active_tracks:
            if track.frame and track.frame[-1] == frame_no:
                # Convert from [xc, yc, ar, h] to xyxy
                # Get the measurement data
                measurement = track.measurement[-1]
                center_x, center_y, aspect_ratio, height = measurement

                # Re-calculate the width from the aspect ratio
                width = aspect_ratio * height

                # Convert to xyxy
                x1 = center_x - (width / 2)
                y1 = center_y - (height / 2)
                x2 = center_x + (width / 2)
                y2 = center_y + (height / 2)

                track_id = track.id
                conf = 1.0
                class_id = 3

                formatted_tracks.append([x1, y1, x2, y2, track_id, conf, class_id])

        return np.array(formatted_tracks)
