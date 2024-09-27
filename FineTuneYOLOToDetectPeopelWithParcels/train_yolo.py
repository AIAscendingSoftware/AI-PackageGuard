from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()  # Only required if you are building an executable; safe to include otherwise

    # Load the pre-trained YOLOv8 model
    model = YOLO('yolov8x.pt')  # Using the yolov8x model

    # Fine-tune the model on the custom dataset
    model.train(
        data=r'D:\AI Projects\building action recognition model\AI-PackageGuard\FineTuneYOLOToDetectPeopelWithParcels\configs\holding_parcel.yaml',  # Path to your data YAML file
        epochs=100,                       # Increase if you want longer training
        batch=8,                         # Adjust based on GPU memory, larger values can stabilize training
        imgsz=640,                        # Resize input images (larger size = better accuracy but slower)
        lr0=0.0005,                       # Learning rate
        name='yolov8_finetuned',          # Run name for experiment logs and saving weights
        freeze=[0],                       # Optionally freeze backbone layers to retain pre-trained knowledge
        patience=10                       # Use early stopping to stop training if it doesn't improve after 10 epochs
    )

    # After training, evaluate the model using the validation/test set
    results = model.val(data=r'D:\AI Projects\building action recognition model\AI-PackageGuard\FineTuneYOLOToDetectPeopelWithParcels\configs\holding_parcel.yaml')

    # Optional: Print the evaluation results
    print('results:', results)

    # Optionally, you can save the evaluation results to a file
    with open('evaluation_results.txt', 'w') as f:
        f.write(str(results))
