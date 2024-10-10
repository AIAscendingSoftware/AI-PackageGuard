# #initializing the classes from other files
# from cnn_lstm_model import CNNLSTMModelArchitecture

# #importing necessary dependencies
# import numpy as np
# import json,os
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau




# class ModelSetup:
#     def __init__(self,X_train, X_test, y_train, y_test,input_shape,num_classes,results_dir):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.X_train=X_train
#         self.X_test=X_test
#         self.y_train=y_train
#         self.y_test=y_test
#         self.results_dir=results_dir

#     def save_plot(self,fig, filename):
#         fig.savefig(filename)
#         plt.close(fig)
        
#     def reshaping(self):
#         # Ensure that the reshaping aligns correctly
#         time_steps=20 #nnumber of frames per video sequence (or segment) that the model will process as input
#         self.X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], time_steps, 224, 224, 3)
#         self.X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], time_steps, 224, 224, 3)

#         print(f"Reshaped X_train shape: {self.X_train_reshaped.shape}")
#         print(f"Reshaped y_train shape: {self.y_train.shape}")
        
#     def Callbacks(self):
#         # Callbacks: Model Checkpoint, Early Stopping, ReduceLROnPlateau (for learning rate adaptation)

#         checkpoint_path = os.path.join(self.results_dir, 'model_checkpoint.h5')
#         self.checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
#         self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
#         self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

#     def Fitmodel(self):
#         self.cnn_lstm_model = CNNLSTMModelArchitecture(self.input_shape, self.num_classes)
#         self.cnn_lstm_model.model.summary()
        
#         # Fit the model
#         self.history = self.cnn_lstm_model.model.fit(
#             self.X_train_reshaped, 
#             self.y_train, 
#             validation_data=(self.X_test_reshaped, self.y_test), 
#             epochs=100, 
#             batch_size=32, 
#             callbacks=[self.checkpoint, self.early_stopping,self.reduce_lr ]
#         )
        
#     def savefinalmodel(self):

#         # Save the final model
#         final_model_path = os.path.join(self.results_dir, 'final_cnn_lstm_model.h5')
#         self.cnn_lstm_model.model.save(final_model_path)

#     def evaluatemodel(self):

#         # Evaluate the model
#         y_pred = self.cnn_lstm_model.model.predict(self.X_test_reshaped)
#         self.y_pred_classes = np.argmax(y_pred, axis=1)

#         # Convert y_test to integer labels if it's one-hot encoded
#         if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1:  # Check if y_test is one-hot encoded
#             self.y_test_int = np.argmax(self.y_test, axis=1)
#         else:
#             self.y_test_int = self.y_test  # Already integer labels

#         # Classification Report
#         class_report = classification_report(self.y_test_int, self.y_pred_classes, target_names=['Throwing', 'Handling', 'Sitting'], output_dict=True)

#         # Save classification report as JSON
#         with open(os.path.join(self.results_dir, 'classification_report.json'), 'w') as f:
#             json.dump(class_report, f, indent=4)

#     # Confusion Matrix
#     def plotsaveconfusionmatrix(self):
        
#         cm = confusion_matrix(self.y_test_int, self.y_pred_classes)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Throwing', 'Handling', 'Sitting'], yticklabels=['Throwing', 'Handling', 'Sitting'])
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')
#         self.save_plot(plt.gcf(), os.path.join(self.results_dir, 'confusion_matrix.png'))

#     # Plot training history
#     def plotsavetraininghistory(self):
        
#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 2, 1)
#         plt.plot(self.history.history['loss'], label='train loss')
#         plt.plot(self.history.history['val_loss'], label='val loss')
#         plt.title('Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.subplot(1, 2, 2)
#         plt.plot(self.history.history['accuracy'], label='train accuracy')
#         plt.plot(self.history.history['val_accuracy'], label='val accuracy')
#         plt.title('Accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         self.save_plot(plt.gcf(), os.path.join(self.results_dir, 'training_history.png'))
    
#     def savetraininghistory(self):
#         # Save training history as JSON
#         with open(os.path.join(self.results_dir, 'training_history.json'), 'w') as f:
#             json.dump(self.history.history, f, indent=4)

#         print(f"All results have been saved in the '{self.results_dir}' directory.")

# Initializing the classes from other files
from cnn_lstm_model import CNNLSTMModelArchitecture

# Importing necessary dependencies
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class ModelSetup:
    def __init__(self, X_train, X_test, y_train, y_test, input_shape, num_classes, results_dir):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results_dir = results_dir

    def save_plot(self, fig, filename):
        fig.savefig(filename)
        plt.close(fig)

    def reshaping(self):
        # Ensure that the reshaping aligns correctly
        self.X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], 20, 224, 224, 3)
        self.X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], 20, 224, 224, 3)

        print(f"Reshaped X_train shape: {self.X_train_reshaped.shape}")
        print(f"Reshaped y_train shape: {self.y_train.shape}")

    def Callbacks(self):
        # Callbacks
        checkpoint_path = os.path.join(self.results_dir, 'model_checkpoint.h5')
        self.checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        # self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

    def Fitmodel(self):
        self.cnn_lstm_model = CNNLSTMModelArchitecture(self.input_shape, self.num_classes)
        self.cnn_lstm_model.model.summary()
        
        # Fit the model
        self.history = self.cnn_lstm_model.model.fit(
            self.X_train_reshaped, 
            self.y_train, 
            validation_data=(self.X_test_reshaped, self.y_test), 
            epochs=100, 
            batch_size=8, 
            callbacks=[self.checkpoint, self.early_stopping]
        )
        
    def savefinalmodel(self):
        # Save the final model
        final_model_path = os.path.join(self.results_dir, 'final_cnn_lstm_model.h5')
        self.cnn_lstm_model.model.save(final_model_path)

    def evaluatemodel(self):
        # Evaluate the model
        y_pred = self.cnn_lstm_model.model.predict(self.X_test_reshaped)
        self.y_pred_classes = np.argmax(y_pred, axis=1)

        # Convert y_test to integer labels if it's one-hot encoded
        if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1:  # Check if y_test is one-hot encoded
            self.y_test_int = np.argmax(self.y_test, axis=1)
        else:
            self.y_test_int = self.y_test  # Already integer labels

        # Classification Report
        class_report = classification_report(self.y_test_int, self.y_pred_classes, target_names=['Throwing', 'Handling', 'Sitting'], output_dict=True)

        # Save classification report as JSON
        with open(os.path.join(self.results_dir, 'classification_report.json'), 'w') as f:
            json.dump(class_report, f, indent=4)

    # Confusion Matrix
    def plotsaveconfusionmatrix(self):
        cm = confusion_matrix(self.y_test_int, self.y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Throwing', 'Handling', 'Sitting'], yticklabels=['Throwing', 'Handling', 'Sitting'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        self.save_plot(plt.gcf(), os.path.join(self.results_dir, 'confusion_matrix.png'))

    # Plot training history
    def plotsavetraininghistory(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='train loss')
        plt.plot(self.history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='train accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        self.save_plot(plt.gcf(), os.path.join(self.results_dir, 'training_history.png'))
    
    def savetraininghistory(self):
        # Save training history as JSON
        with open(os.path.join(self.results_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history.history, f, indent=4)

        print(f"All results have been saved in the '{self.results_dir}' directory.")
