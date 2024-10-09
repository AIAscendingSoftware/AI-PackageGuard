###use python version 3.12 for both runnung the fine-tuned yolo model and creating the cnn + lstm architecture

###To activatge the GPU

we ahve NVIDIA 
    CUDA version is 12.5 in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA path
    cuDNN version is 8.9.7 in "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\include\cudnn_version.h"
For these systems, wwe need to use Tensorflow 2.14 and
    Recommended Python Versions for TensorFlow 2.14.0:
    3.8
    3.9
    3.10

finally after several testing, we are going to use 'conda_venv_for_GPU' environment rather normal python environment, because of for us conda virtual environment supports tensorflow to run the training process on CUDA+cuDNN