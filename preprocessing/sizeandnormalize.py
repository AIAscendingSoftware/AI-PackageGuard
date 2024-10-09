import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from PIL import Image



# class DataPreprocessor:
#     def __init__(self, target_length, target_size=(224, 224)):
#         self.target_length = target_length
#         self.target_size = target_size

#     def preprocess_data(self, padded_dataset_folder, action_labels):
#         X, y = [], []

#         for action in action_labels:
#             action_folder = os.path.join(padded_dataset_folder, action)
#             for sequence in os.listdir(action_folder):
#                 sequence_folder = os.path.join(action_folder, sequence)
#                 frames = []

#                 for frame_name in sorted(os.listdir(sequence_folder)):
#                     frame_path = os.path.join(sequence_folder, frame_name)
#                     # Resize and normalize image
#                     image = self.resize_and_normalize(frame_path)
#                     frames.append(image)

#                 # Only take sequences of target length
#                 if len(frames) == self.target_length:
#                     X.append(frames)
#                     y.append(action_labels.index(action))  # Numeric label for the action
#                 else:
#                     print(f"Sequence '{sequence}' has {len(frames)} frames (not added).")

#         # Debug: Print shapes of frames in X before conversion
#         for idx, seq in enumerate(X):
#             print(f"Shape of sequence {idx}: {np.array(seq).shape}")

#         # Convert to numpy arrays
#         try:
#             X = np.array(X)
#             y = np.array(y)  # Convert y to a numpy array before reshaping
#             print(f"X shape: {X.shape}")  # Print the shape of X after conversion
#             print(f"y length: {len(y)}")   # Print the length of y

#             # One-hot encode labels
#             y_one_hot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

#             # Split the dataset
#             return train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
#         except ValueError as e:
#             print(f"Error converting to array: {e}")
#             print(f"X shape: {[len(seq) for seq in X]}")  # Print lengths of sequences
#             return None, None, None, None  # Return None in case of error

#     def resize_and_normalize(self, image_path):
#         image = Image.open(image_path)
#         image = image.resize(self.target_size)
#         image_array = np.array(image)
#         return image_array / 255.0  # Normalize to [0, 1]

from sklearn.preprocessing import OneHotEncoder
# Other imports remain the same

class DataPreprocessor:
    def __init__(self, target_length, target_size=(224, 224)):
        self.target_length = target_length
        self.target_size = target_size

    def preprocess_data(self, padded_dataset_folder, action_labels):
        X, y = [], []

        for action in action_labels:
            action_folder = os.path.join(padded_dataset_folder, action)
            if not os.path.exists(action_folder):
                print(f"Action folder '{action_folder}' does not exist.")
                continue

            for sequence in os.listdir(action_folder):
                sequence_folder = os.path.join(action_folder, sequence)
                frames = []

                for frame_name in sorted(os.listdir(sequence_folder)):
                    frame_path = os.path.join(sequence_folder, frame_name)
                    # Resize and normalize image
                    image = self.resize_and_normalize(frame_path)
                    frames.append(image)

                # Only take sequences of target length
                if len(frames) == self.target_length:
                    X.append(frames)
                    y.append(action_labels.index(action))  # Numeric label for the action
                else:
                    print(f"Sequence '{sequence}' has {len(frames)} frames (not added).")

        # Debug: Print shapes of frames in X before conversion
        for idx, seq in enumerate(X):
            print(f"Shape of sequence {idx}: {np.array(seq).shape}")

        # Convert to numpy arrays
        try:
            X = np.array(X)
            y = np.array(y)  # Convert y to a numpy array before reshaping
            print(f"X shape: {X.shape}")  # Print the shape of X after conversion
            print(f"y length: {len(y)}")   # Print the length of y

            # One-hot encode labels
            y_one_hot = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()  # Converting sparse matrix to dense

            # Split the dataset
            return train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
        except ValueError as e:
            print(f"Error converting to array: {e}")
            print(f"X shape: {[len(seq) for seq in X]}")  # Print lengths of sequences
            return None, None, None, None  # Return None in case of error


    def resize_and_normalize(self, image_path):
        image = Image.open(image_path)
        image = image.resize(self.target_size)
        image_array = np.array(image)
        return image_array / 255.0  # Normalize to [0, 1]
