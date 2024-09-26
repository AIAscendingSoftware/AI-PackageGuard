---Virtual environment---
we are going to use virtual environment at "E:\AI Ascending Software\AS AI Projects\yolo\venvForYolo" which is used for fine tuning the yolo for detecting throwing parcels

step1: run the 'convertVideoToFrames.py' files by giving the videos containing path to extract frames from the videos to store on 'extractedframesfromvideos' folder

step2: look at the 'extractedframesfromvideos' folder, and bring the parcels holding frames only to 'holdingparcelsframes' folder

step3: run the 'splitingDataset.py' file by giving necessary folder path to save the splited images from 'holdingparcelsframes' folder

step4: go to the labelImg tool in the necessary tools folder to annotate each frames in the train, test, val folders frames and keep proper lables and and save the txt annotated points on necessaary folder like labels\train,test,val while doing manual annotation on bounding box make sure we've enabled the yolo

step5: note on 'imagesToRemove.txt' file what are the frames need to be deleted while annotating bounding box

step6: run 'validateDatasetsAndAnnotation.py' file by giving the custom_dataset path, this automation code will tell what are the img files don't have txt pair file which has the data point on <object_class> <x_center> <y_center> <width> <height> format, and where ther are no proper img files in the image and label inside folders, then we need to remove unnessary files, then make sure by rerunnig the 'validateDatasetsAndAnnotation.py' file which doesn't return anything, tha means, all frames are having their proper txt pair files

step7: run 'train_yolo.py' by giving yaml path, epochs, batch, image size, and name, we should keep 'yolov8x.pt' on local, because of we are going to fine-tune it, one we've successfully fine-tune, it will generate the 'yolov8_finetuned' which has 'best.pt', and 'last.pt' models, then we have to use the best model

