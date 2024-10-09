import cv2
from ultralytics import YOLO

# Load the fine-tuned YOLO model
model = YOLO(r'E:\AI Ascending Software\AS AI Projects\building action recognition model\detecting holding parcels model\weights\best.pt')  # Path to your fine-tuned model

# Open video feed
video_path = r"E:\AI Ascending Software\AS AI Projects\yolo\videos\WhatsApp Video 2024-08-30 at 21.13.02 (2).mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection using the fine-tuned model
    results = model(frame)
    
    # Draw results on the frame
    annotated_frame = results[0].plot()  # Draw bounding boxes
    
    # Display the frame
    cv2.imshow('Live Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


