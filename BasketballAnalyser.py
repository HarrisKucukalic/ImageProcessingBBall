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
        self.stats_recorder = StatsRecorder()

    def _draw_tracks(self, frame, tracks):
        """Draws bounding boxes, track IDs, and player trails on the frame."""
        # Draw player trails first (so they are underneath the boxes)
        for player in self.stats_recorder.players.values():
            if len(player.positions) > 1:
                for i in range(1, len(player.positions)):
                    if player.positions[i - 1] is None or player.positions[i] is None:
                        continue
                    # Calculate thickness based on position in the deque for a fading effect
                    thickness = int(np.sqrt(10 * float(i + 1) / len(player.positions)) * 2)
                    cv2.line(frame, player.positions[i - 1], player.positions[i], (0, 255, 255), thickness)

        # Draw bounding boxes and labels
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, conf = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id, class_id = int(track_id), int(class_id)

            class_name = CLASS_NAMES.get(class_id, 'unknown')
            color = CLASS_COLORS.get(class_name, (255, 255, 255))

            label = f"P{track_id}" if class_name == 'player' else class_name

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def process_video(self):
        """Main processing loop for the video."""
        frame_count = 0
        start_time = time.time()

        print(f"Starting tracking with '{self.tracker_config}' on '{self.video_source}'...")
        results_generator = self.model.track(
            source=self.video_source,
            tracker=self.tracker_config,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            stream=True,
            verbose=False,
            # Maintains tracks between frames
            persist=True
        )

        for result in results_generator:
            frame = result.orig_img
            annotated_frame = frame.copy()

            tracks = []
            # Check if there are any tracked objects
            if result.boxes.id is not None:
                # The result.boxes.data tensor is in the format: [x1, y1, x2, y2, track_id, conf, class_id]
                detections_with_ids = result.boxes.data.cpu().numpy()

                # Reformat for StatsRecorder: swap conf and class_id to get [..., track_id, class_id, conf]
                # Ensure it has all expected columns
                if detections_with_ids.shape[1] == 7:

                    tracks = np.column_stack((
                        detections_with_ids[:, 0:4],  # x1, y1, x2, y2
                        detections_with_ids[:, 4],    # track_id
                        detections_with_ids[:, 6],    # class_id
                        detections_with_ids[:, 5]     # conf
                    ))

                    # Update statistics module
                    self.stats_recorder.update(tracks)

                    # Draw tracking overlays
                    annotated_frame = self._draw_tracks(annotated_frame, tracks)

            # Draw the stats overlay regardless of whether there are tracks in the current frame
            annotated_frame = self.stats_recorder.draw_stats_overlay(annotated_frame)

            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Basketball Analyzer", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Processing complete.")