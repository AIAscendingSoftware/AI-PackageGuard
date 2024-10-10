

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

class CNNLSTMModelArchitecture:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes) 

    def build_model(self, input_shape, num_classes):
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(32))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

# from keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Bidirectional, LSTM, Dense, Dropout, Attention, BatchNormalization 
# from keras.models import Model
# from keras.regularizers import l2

# class CNNLSTMModelArchitecture:
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.model = self.build_model(input_shape, num_classes)

#     def build_model(self, input_shape, num_classes):
#         # Define the input layer
#         inputs = Input(shape=input_shape)

#         # First CNN block with multi-scale feature extraction
#         x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
#         x = TimeDistributed(Conv2D(32, (5, 5), activation='relu', padding='same'))(x)
#         x = TimeDistributed(Conv2D(32, (7, 7), activation='relu', padding='same'))(x)
#         x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
#         x = TimeDistributed(BatchNormalization())(x)

#         # Second CNN block
#         x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
#         x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
#         x = TimeDistributed(BatchNormalization())(x)

#         # Third CNN block
#         x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
#         x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
#         x = TimeDistributed(BatchNormalization())(x)

#         # Flatten CNN output before feeding into RNN layers
#         x = TimeDistributed(Flatten())(x)

#         # Bidirectional LSTM
#         lstm_output, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(x)
        
#         # Adding Attention layer
#         attention_output = Attention()([lstm_output, lstm_output])  # Use the output as both query and value

#         # Final LSTM layer after attention
#         x = Bidirectional(LSTM(32))(attention_output)

#         # Dropout layer
#         x = Dropout(0.5)(x)

#         # Output Layer
#         outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

#         # Create model
#         model = Model(inputs, outputs)

#         # Compile model
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#         return model
