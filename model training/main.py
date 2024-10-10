#initializing the classes from other files
from sizeandnormalize import DataPreprocessor
from model_setup import ModelSetup


#
import torch,json,os,logging
import matplotlib.pyplot as plt
from datetime import datetime



# Set environment variable to use CPU (for debugging)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



if __name__ == "__main__":
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a directory for saving results
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # Data Preprocessing
    data_preprocessor = DataPreprocessor(target_length=20)
    X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data(
        r'D:\AI Projects\building action recognition model\AI-PackageGuard\padded_dataset',
        ['throwing_parcels', 'handling_parcels_properly', 'sitting_on_parcels']
    )
    print('X_test shape:', X_test.shape, 'y_test shape:', y_test.shape)
    print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape)


    # Model Setup
    input_shape = (20, 224, 224, 3)  # 20 frames, 224x224 image size, 3 channels
    num_classes = 3  # Number of action classes
    Model_Setup=ModelSetup(X_train, X_test, y_train, y_test,input_shape,num_classes,results_dir)
    Model_Setup.reshaping()
    Model_Setup.Callbacks()
    Model_Setup.Fitmodel()
    Model_Setup.savefinalmodel()
    Model_Setup.evaluatemodel()
    Model_Setup.plotsaveconfusionmatrix()
    Model_Setup.plotsavetraininghistory()
    Model_Setup.savetraininghistory()
