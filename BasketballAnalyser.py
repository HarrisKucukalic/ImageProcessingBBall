from ultralytics import YOLO
from StatsRecorder import *

CLASS_COLORS = {
    'player': (0, 255, 0),       # Green
    'basketball': (255, 165, 0),  # Orange
    'rim': (0, 0, 255)            # Red
}


class BasketballAnalyser:
    """
    Main class to orchestrate the detection, tracking, and analysis process.
    """

    def __init__(self, model_path, video_source, tracker_config='bytetrack.yaml', conf_thresh=0.5, iou_thresh=0.7):
        # Initialise YOLO model
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        # Store video source and config
        self.video_source = video_source
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # Initialise statistics recorder
        self.stats_recorder = None

    def _draw_tracks(self, frame, tracks):
        """Draws dots and track IDs on the frame instead of bounding boxes."""
        # Draw player dots and labels
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, conf = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id, class_id = int(track_id), int(class_id)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            class_name = CLASS_NAMES.get(class_id, 'unknown')
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            label = f"P-{track_id}" if class_id == PLAYER_CLASS_ID else class_name
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            cv2.putText(frame, label, (center_x - 10, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def _draw_stats_panel(self, frame):
        """Draws the possession stats panel on the frame without any trails."""
        if not self.stats_recorder:
            return frame

        # Draw a semi-transparent box for the stats
        box_x, box_y, box_w, box_h = 10, 10, 300, 100
        sub_img = frame[box_y:box_y + box_h, box_x:box_x + box_w]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 1.0)
        frame[box_y:box_y + box_h, box_x:box_x + box_w] = res

        # Draw border
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1, cv2.LINE_AA)

        # Display possession info
        text_y = box_y + 25
        cv2.putText(frame, "Possession Stats", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        text_y += 35
        player_id = self.stats_recorder.player_with_ball_id
        if player_id is not None and player_id in self.stats_recorder.player_stats:
            poss_time = self.stats_recorder.player_stats[player_id].possession_time
            poss_text = f"Player {player_id}: {poss_time:.1f}s"
            cv2.putText(frame, poss_text, (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No possession", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame

    def process_video(self):
        """
        Processes the video source, performs object tracking, and displays the results.
        """
        start_time = time.time()
        frame_count = 0

        try:
            results_generator = self.model.track(
                source=self.video_source,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                tracker=self.tracker_config,
                persist=True,
                stream=True,
                show=False,
                verbose=False
            )

            is_recorder_initialized = False

            for results in results_generator:
                frame = results.orig_img
                annotated_frame = frame.copy()

                if not is_recorder_initialized:
                    video_fps = 30  # Default FPS
                    if hasattr(results, 'fps') and results.fps:
                        video_fps = results.fps
                    self.stats_recorder = StatsRecorder(video_fps)
                    is_recorder_initialized = True

                if results.boxes.id is not None:
                    # The result.boxes.data tensor is [x1, y1, x2, y2, track_id, conf, class_id]
                    detections_with_ids = results.boxes.data.cpu().numpy()

                    # Format tracks for StatsRecorder.update which expects list of (box, track_id, cls_id)
                    tracks_for_stats = [((d[0:4]), int(d[4]), int(d[6])) for d in detections_with_ids]
                    self.stats_recorder.update(tracks_for_stats)

                    # Format tracks for _draw_tracks which expects list of (x1, y1, x2, y2, track_id, class_id, conf)
                    tracks_for_drawing = [(d[0], d[1], d[2], d[3], d[4], d[6], d[5]) for d in detections_with_ids]

                    # Use the custom drawing function instead of results.plot()
                    annotated_frame = self._draw_tracks(annotated_frame, tracks_for_drawing)

                annotated_frame = self._draw_stats_panel(annotated_frame)

                # Calculate and display FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, annotated_frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Basketball Analysis", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("'q' pressed, stopping video processing.")
                    break

        except Exception as e:
            print(f"An error occurred during video processing: {e}")
        finally:
            cv2.destroyAllWindows()
            print("Video processing finished and windows closed.")