
'''intracking while handlg parcel with deepsort'''

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class ParcelTracker:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=50, n_init=3, nn_budget=100, max_iou_distance=0.9)
        self.video_path = video_path
        self.cap = None

    def open_video(self):
        self.cap = cv2.VideoCapture(self.video_path)

    def close_video(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def detect_objects(self, frame):
        results = self.model(frame)
        detections = results[0]
        detection_list = []

        for result in detections.boxes:
            box = result.xyxy.cpu().numpy()[0]
            confidence = float(result.conf.cpu().numpy()[0])
            label = int(result.cls.cpu().numpy()[0])
            class_name = detections.names[label]

            if class_name == 'holding parcel' and confidence > 0.60:
                x1, y1, x2, y2 = map(float, box)
                detection_list.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'holding parcel'))

        return detection_list

    def update_tracks(self, detection_list, frame):
        return self.tracker.update_tracks(detection_list, frame=frame)

    def annotate_frame(self, frame, tracks):
        for track in tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                track_id = track.track_id
                ltrb = track.to_ltrb()

                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def process_video(self):
        self.open_video()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            detection_list = self.detect_objects(frame)
            tracks = self.update_tracks(detection_list, frame)
            annotated_frame = self.annotate_frame(frame, tracks)

            cv2.imshow('Tracking Holding Parcel', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.close_video()

def main():
    model_path = r'E:\AI Ascending Software\AS AI Projects\building action recognition model\detecting holding parcels model\weights\best.pt'
    video_path = r"E:\AI Ascending Software\AS AI Projects\building action recognition model\AI-PackageGuard\data\raw_videos\1.mp4"
    
    tracker = ParcelTracker(model_path, video_path)
    tracker.process_video()

if __name__ == "__main__":
    main()