import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
class CNNLSTMModel:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        model = Sequential()

        # TimeDistributed CNN layers
        model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Flatten()))

        # LSTM layers
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(32))

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
