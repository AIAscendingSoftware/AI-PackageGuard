

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image

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
                    image = self.resize_and_normalize(frame_path)
                    frames.append(image)

                if len(frames) == self.target_length:
                    X.append(frames)
                    y.append(action_labels.index(action))

        print(f"Total sequences processed: {len(X)}")
        X = np.array(X)
        y = np.array(y)

        # One-hot encode labels
        y_one_hot = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        return train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    def resize_and_normalize(self, image_path):
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
        image = image.resize(self.target_size)
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        return image_array
