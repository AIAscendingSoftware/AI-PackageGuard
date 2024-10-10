# import tensorflow as tf
# print(tf.__version__)

# # # Check if GPU is available
# # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# import torch
# print(torch.cuda.is_available())  # Should return True if GPU is available
# print(torch.cuda.device_count())  # Should return number of available GPUs

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# import tensorflow as tf
# from keras import layers, models

# # Example of defining a simple model
# def create_model():
#     model = models.Sequential()
#     model.add(layers.Input(shape=(20, 224, 224, 3)))
#     model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
#     # Add other layers...
#     model.add(layers.Dense(3, activation='softmax'))  # Assuming 3 classes
#     return model

# # Check if GPU is available
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# # Create and compile the model
# model = create_model()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Check model summary
# model.summary()
# Simple tensor operation to test GPU
import tensorflow as tf

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
else:
    print("No GPUs available.")

# Simple tensor operation to test GPU
with tf.device('/GPU:0'):  # Specify the GPU device
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.add(a, b)

print("Result of addition on GPU:", c.numpy())
