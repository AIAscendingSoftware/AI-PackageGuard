
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# Load the fine-tuned YOLO model
model = YOLO(r'E:\AI Ascending Software\AS AI Projects\building action recognition model\detecting holding parcels model\weights\best.pt')

# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_iou_distance=0.7)

# Open video feed
video_path = r"E:\AI Ascending Software\AS AI Projects\yolo\videos\WhatsApp Video 2024-08-30 at 21.13.02 (2).mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection using the fine-tuned YOLO model
    results = model(frame)
    detections = results[0]  # Get detections from YOLO model
    
    # Prepare detection data for DeepSORT
    detection_list = []
    for result in detections.boxes:
        box = result.xyxy.cpu().numpy()[0]  # Extract bounding box [x1, y1, x2, y2]
        confidence = float(result.conf.cpu().numpy()[0])  # Convert confidence to float
        label = int(result.cls.cpu().numpy()[0])  # Class ID (should be 'holding parcel')
        class_name = detections.names[label]  # Get the label name
        
        # Filter for 'holding parcel' detections with confidence above 0.80
        if class_name == 'holding parcel' and confidence > 0.80:
            x1, y1, x2, y2 = map(float, box)
            
            # DeepSORT expects: (left, top, width, height, confidence)
            detection_list.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'holding parcel'))
    
    # Update DeepSORT with filtered YOLO detections
    if detection_list:
        tracks = tracker.update_tracks(detection_list, frame=frame)  # Get the tracked objects
        
        # Annotate the frame with DeepSORT tracking results
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Bounding box in (left, top, right, bottom) format
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Live Detection and Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()