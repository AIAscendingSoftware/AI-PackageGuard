
from sizeandnormalize import DataPreprocessor
from cnn_lstm_model import CNNLSTMModel

# # Example usage

# if __name__ == "__main__":
#     data_preprocessor = DataPreprocessor(target_length=20)
#     X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data(
#         r'D:\AI Projects\building action recognition model\AI-PackageGuard\padded_dataset',
#         ['throwing_parcels', 'handling_parcels_properly', 'sitting_on_parcels']
#     )
#     print('X_test shape:', X_test.shape, 'y_test shape:', y_test.shape)
#     print('X_test shape:', X_train.shape, 'y_test shape:', y_train.shape)


#     # Assuming input shape based on your data
#     input_shape = (20, 224, 224, 3)  # 20 frames, 224x224 image size, 3 channels
#     num_classes = 3  # Number of action classes

#     cnn_lstm_model = CNNLSTMModel(input_shape, num_classes)

#     # Print model summary
#     cnn_lstm_model.model.summary()

#     # Fit the model using your training data
#     # X_train, X_test, y_train, y_test should be loaded from your previous preprocessing step
#     cnn_lstm_model.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

import numpy as np
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Example usage
if __name__ == "__main__":
    data_preprocessor = DataPreprocessor(target_length=20)
    X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data(
        r'D:\AI Projects\building action recognition model\AI-PackageGuard\padded_dataset',
        ['throwing_parcels', 'handling_parcels_properly', 'sitting_on_parcels']
    )
    print('X_test shape:', X_test.shape, 'y_test shape:', y_test.shape)
    print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape)

    # Assuming input shape based on your data
    input_shape = (20, 224, 224, 3)  # 20 frames, 224x224 image size, 3 channels
    num_classes = 3  # Number of action classes

    cnn_lstm_model = CNNLSTMModel(input_shape, num_classes)

    # Print model summary
    cnn_lstm_model.model.summary()

    # Ensure that the reshaping aligns correctly
    X_train_reshaped = X_train.reshape(X_train.shape[0], 20, 224, 224, 3)  # This should be 77 samples
    X_test_reshaped = X_test.reshape(X_test.shape[0], 20, 224, 224, 3)  # This should be 20 samples

    print(f"Reshaped X_train shape: {X_train_reshaped.shape}")
    print(f"Reshaped y_train shape: {y_train.shape}")

    # Fit the model
    cnn_lstm_model.model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=50, batch_size=32)
