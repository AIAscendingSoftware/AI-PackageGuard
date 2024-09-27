---Virtual environment---
we are going to use virtual environment at "E:\AI Ascending Software\AS AI Projects\yolo\venvForYolo" which is used for fine tuning the yolo for detecting throwing parcels

---Preprocessing---

  Initial step:

  Preprocessing:
      use this link to trim a video:https://tools.invideo.io/tools/video-trimmer/

      ############
      For YOLOv8x (the extra-large variant of YOLOv8), the default input images size is 640x640 pixels.

      ############
      use this like :https://github.com/HumanSignal/labelImg/releases to download LableImg
      which helps to annotate the images which has availability to get YOLO annoatation marks

      ###########
      dataset should be organized as follows:

      custom_dataset/
      │
      ├── images/
      │   ├── train/
      │   │   ├── img1.jpg
      │   │   ├── img2.jpg
      │   │   └── ...
      │   ├── val/
      │   │   ├── img1.jpg
      │   │   ├── img2.jpg
      │   │   └── ...
      │   └── test/  # Optional, for testing only
      │       ├── img1.jpg
      │       ├── img2.jpg
      │       └── ...
      │
      ├── labels/
      │   ├── train/
      │   │   ├── img1.txt
      │   │   ├── img2.txt
      │   │   └── ...
      │   ├── val/
      │   │   ├── img1.txt
      │   │   ├── img2.txt
      │   │   └── ...
      │   └── test/  # Optional, for testing only
      │       ├── img1.txt
      │       ├── img2.txt
      │       └── ...

      ######## To activate GPU ########
      we use python 3.9 to fine-tune the yolo, so to utilize the GPU, install necessary libraries with corresponding versionb as mentioned below

      pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

  step1: run the 'convertVideoToFrames.py' files by giving the videos containing path to extract frames from the videos to store on 'extractedframesfromvideos' folder
  
  step2: look at the 'extractedframesfromvideos' folder, and bring the parcels holding frames only to 'holdingparcelsframes' folder

  step3: run the 'splitingDataset.py' file by giving necessary folder path to save the splited images from 'holdingparcelsframes' folder

  step4: go to the labelImg tool in the necessary tools folder to annotate each frames in the train, test, val folders frames and keep proper lables and and save the txt annotated points on necessaary folder like labels\train,test,val while doing manual annotation on bounding box make sure we've enabled the yolo

  step5: note on 'imagesToRemove.txt' file what are the frames need to be deleted while annotating bounding box

  step6: run 'validateDatasetsAndAnnotation.py' file by giving the custom_dataset path, this automation code will tell what are the img files don't have txt pair file which has the data point on <object_class> <x_center> <y_center> <width> <height> format, and where ther are no proper img files in the image and label inside folders, then we need to remove unnessary files, then make sure by rerunnig the 'validateDatasetsAndAnnotation.py' file which doesn't return anything, tha means, all frames are having their proper txt pair files

---Fine-Tunning and Evaluation---
  
  step7: run 'train_yolo.py' by giving yaml path, epochs, batch, image size, and name, we should keep 'yolov8x.pt' on local, because of we are going to fine-tune it, one we've successfully fine-tune, it will generate the 'yolov8_finetuned' which has 'best.pt', and 'last.pt' models in the weights folder, then we can use the best model

  step8: the "evaluation_results.txt' will be saved which has evaluation result
  
  step9: things to notice while training the model, here we've shown the action while while training the model 

    Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    9/100      6.94G     0.7621     0.5599     0.9934         18        640: 100%|██████████| 91/91 [00
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████
                       all         96        160      0.804      0.869      0.872      0.668
    
    Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    10/100      6.93G     0.7774     0.5531      1.009          8        640: 100%|██████████| 91/91 [00
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████
                       all         96        160      0.769      0.915        0.9      0.705
    
    Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    11/100      6.94G     0.7712      0.558      1.021         10        640: 100%|██████████| 91/91 [00
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████
                       all         96        160      0.771      0.912      0.886      0.698
    
    Loss Values:
    
    box_loss: Bounding box regression loss. Lower values indicate better bounding box predictions. This should decrease over time.
    cls_loss: Classification loss. This measures how well the model classifies objects into the correct class (parcel/no parcel, etc.). This should also decrease over time.
    dfl_loss: Distributional Focal Loss, used to optimize the bounding box's localization. It should decrease over time too
    
    Validation Metrics (after each epoch): 
    Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
      all         96        160      0.763      0.894      0.831      0.645
    
    Box(P): Precision of the bounding boxes (0.763 or 76.3%). This tells you how many of the predicted bounding boxes were correct.
    R: Recall (0.894 or 89.4%), which tells you how many true objects in the validation dataset were detected by the model.
    mAP50: Mean Average Precision at IoU threshold 0.50 (0.831 or 83.1%). This is a key metric that evaluates both precision and recall over all object classes at a specific intersection over union (IoU) threshold.
    mAP50-95: Mean Average Precision at multiple IoU thresholds between 0.50 and 0.95 (0.645 or 64.5%). A more difficult metric, since it averages the precision across multiple thresholds.
    
    What to Monitor While Training:
    
    Loss Values:
    Box, classification, and DFL losses should decrease over time.
    If the losses stop decreasing for several epochs or start increasing, it might indicate that the model has overfitted or plateaued.
    
    Validation Metrics:
    Precision (P): Should generally increase. It tells how many of the model's predictions are correct.
    Recall (R): Should increase as well, indicating the model is finding more of the true objects in the images.
    mAP50 and mAP50-95: These should also increase over time, indicating that the model's overall detection performance is improving.

    Result after fine-tuning process competed:

    fitness: 0.6786234608372472
    keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    maps: array([    0.65919])
    names: {0: 'holding parcel'}
    plot: True
    results_dict: {'metrics/precision(B)': 0.7587871774225777, 'metrics/recall(B)': 0.9125, 'metrics/mAP50(B)': 0.8534929400605744, 'metrics/mAP50-95(B)': 0.    659193518701322, 'fitness': 0.6786234608372472}     
    save_dir: WindowsPath('runs/detect/yolov8_finetuned2')
    speed: {'preprocess': 0.26041269302368164, 'inference': 25.4964679479599, 'loss': 0.0, 'postprocess': 0.5328158537546793}
    task: 'detect'

  step10: If we wanna retain yolo with it's past knowledge, we should've included yolo's trained dataset(coco dataset) with our custom dataset, but in our case we are gonna just fine-fune the yolov8x with our custom dataset, we don't want yolo to bring with it's past knowledge


---Testing on real environment---
  
  step11: after training necessarty files will be stored on 'yolov8_finetuned' folder which we mentioned while fine tuning the model and our trained model 'best.pt', and 'last.pt' are in D:\AI Projects\building action recognition model\runs\detect\yolov8_finetuned\weights folder, then run the 'test_finetuned_model.py' file by giving the video path and best.pt path to test the fine-tuned model





