import os,cv2,random
import numpy as np
from glob import glob

class SequenceAugmentor:
    def __init__(self, augment_type):
        self.augment_type = augment_type
        self.params = {}  # Store augmentation parameters to ensure consistency across frames

    def set_augmentation_params(self, frame):
        # Define augmentation parameters only once, and use them for all frames in the sequence
        if self.augment_type == 'rotate':
            self.params['angle'] = random.randint(-15, 15)
        elif self.augment_type == 'flip':
            self.params['flip_code'] = 1  # Horizontal flip
        elif self.augment_type == 'brightness':
            self.params['brightness_value'] = random.randint(-30, 30)
        elif self.augment_type == 'shift':
            h, w = frame.shape[:2]
            self.params['tx'] = random.randint(-w//10, w//10)
            self.params['ty'] = random.randint(-h//10, h//10)
        elif self.augment_type == 'zoom':
            self.params['scale'] = random.uniform(0.9, 1.1)
        elif self.augment_type == 'contrast':
            self.params['alpha'] = random.uniform(0.7, 1.3)
        elif self.augment_type == 'blur':
            self.params['ksize'] = random.choice([3, 5])
        elif self.augment_type == 'noise':
            self.params['noise_level'] = 25  # Custom noise level
        print('self.params:',self.params)

    def augment_frame(self, frame):
        # Spatial augmentations (applied consistently to all frames)
        if self.augment_type == 'rotate':
            return self.rotate(frame)
        elif self.augment_type == 'flip':
            return self.flip(frame)
        elif self.augment_type == 'brightness':
            return self.brightness(frame)
        elif self.augment_type == 'shift':
            return self.shift(frame)
        elif self.augment_type == 'zoom':
            return self.zoom(frame)
        elif self.augment_type == 'contrast':
            return self.contrast(frame)
        elif self.augment_type == 'blur':
            return self.blur(frame)
        elif self.augment_type == 'noise':
            return self.add_noise(frame)
        return frame

    def rotate(self, frame):
        angle = self.params['angle'] 
        print('angle:',angle)
        h, w = frame.shape[:2]
        print('h, w :',h, w )
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1) 
        '''[[ 0.99939083 -0.0348995   2.40179691] 
        [ 0.0348995   0.99939083 -1.59885258]]'''
        
        return cv2.warpAffine(frame, M, (w, h)) #the new frame array will be there

    def flip(self, frame):
        print("cv2.flip(frame, self.params['flip_code']) :",cv2.flip(frame, self.params['flip_code']) )
        return cv2.flip(frame, self.params['flip_code'])  # Horizontal flip 

    def brightness(self, frame):
        value = self.params['brightness_value']
        return cv2.convertScaleAbs(frame, alpha=1, beta=value)

    def shift(self, frame):
        h, w = frame.shape[:2]
        tx = self.params['tx']
        ty = self.params['ty']
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(frame, M, (w, h))

    def zoom(self, frame):
        scale = self.params['scale']
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
        return cv2.warpAffine(frame, M, (w, h))

    def contrast(self, frame):
        alpha = self.params['alpha']
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    def blur(self, frame):
        ksize = self.params['ksize']
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)

    def add_noise(self, frame):
        noise_level = self.params['noise_level']
        noise = np.random.normal(0, noise_level, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(frame, noise)
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)


class SequenceProcessor:
    def __init__(self, source_sequence_path, new_sequence_path, augmentor):
        self.source_sequence_path = source_sequence_path
        self.new_sequence_path = new_sequence_path
        self.augmentor = augmentor

        if not os.path.exists(new_sequence_path):
            os.makedirs(new_sequence_path)

    def process_sequence(self):
        frames = sorted(glob(os.path.join(self.source_sequence_path, '*.png'))) #all the frames paths would be there

        augmented_sequence = []

        if frames:
            # Set consistent augmentation parameters based on the first frame
            frame = cv2.imread(frames[0])

            self.augmentor.set_augmentation_params(frame)

        for i, frame_path in enumerate(frames):
            print('frame_path:',frame_path)
            frame = cv2.imread(frame_path)
            augmented_frame = self.augmentor.augment_frame(frame)
            # print('augmented_frame:',augmented_frame)
            augmented_sequence.append(augmented_frame)
            print('os.path.basename(frame_path):',os.path.basename(frame_path))
            # Ensure the augmented frame is saved with the original frame's order
            new_frame_path = os.path.join(self.new_sequence_path, os.path.basename(frame_path))
            cv2.imwrite(new_frame_path, augmented_frame)
            print(f"Saving {new_frame_path}...")

# Example usage
source_sequence = r"E:\AI Ascending Software\AS AI Projects\building action recognition model\AI-PackageGuard\dataset\throwing_parcels\sequence_3"
new_sequence = r"E:\AI Ascending Software\AS AI Projects\building action recognition model\AI-PackageGuard\dataset\throwing_parcels\sequence_34"

# Create an augmentor object with the desired augmentation type
augmentor = SequenceAugmentor(augment_type='flip')  # Choose 'rotate', 'flip','brightness','scale','alpha' etc.
# print(augmentor) #<__main__.SequenceAugmentor object at 0x000001B32D6E2E50>
processor = SequenceProcessor(source_sequence, new_sequence, augmentor)
processor.process_sequence()
