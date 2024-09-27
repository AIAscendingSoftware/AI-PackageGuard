
import cv2
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path):
        """
        Initialize the YOLO model with the given model path.
        
        :param model_path: Path to the pre-trained model file (e.g., 'best.pt').
        """
        self.model = YOLO(model_path)  # Load the YOLO model
    
    def run_inference_on_video(self, source):
        """
        Run inference on a video, annotate the frames with predictions, and display the video in real-time.
        
        :param source: Path to the video for inference.
        """
        # Open the video file
        video_cap = cv2.VideoCapture(source)
        
        # Check if the video is opened successfully
        if not video_cap.isOpened():
            print(f"Error: Unable to open video file {source}")
            return
        
        # Get the video width, height, and FPS for display
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        
        # Process the video frame by frame
        while True:
            ret, frame = video_cap.read()  # Read each frame
            if not ret:
                break  # Break if video ends
            
            # Run YOLO inference on the current frame
            results = self.model.predict(frame)  # Predict on the current frame
            
            # Annotate the frame with the prediction results
            annotated_frame = results[0].plot()  # Get the annotated frame
            
            # Display the annotated frame in a video window
            cv2.imshow('YOLO Video Inference', annotated_frame)
            
            # Press 'q' to exit the video window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video capture object and close the display window
        video_cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    # Create an instance of the YOLOModel class with the best model
    yolo_model = YOLOModel(r'D:\AI Projects\building action recognition model\fine-tuned model(holding parcel)\weights\best.pt')
    
    # Run inference on a video and show annotated video in real-time
    yolo_model.run_inference_on_video(r"D:\AI Projects\building action recognition model\AI-PackageGuard\data\raw_videos\2.mp4")