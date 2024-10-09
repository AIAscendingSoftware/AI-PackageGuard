import os
import cv2

def create_image_sequences(folder_path, output_folder, sequence_length=20):
    actions = os.listdir(folder_path)
    for action in actions:
        action_folder = os.path.join(folder_path, action)
        if not os.path.isdir(action_folder):
            continue

        for sequence in os.listdir(action_folder):
            sequence_folder = os.path.join(action_folder, sequence)
            frames = []
            for frame_name in sorted(os.listdir(sequence_folder)):
                frame_path = os.path.join(sequence_folder, frame_name)
                image = cv2.imread(frame_path)
                if image is not None:
                    frames.append(image)
                if len(frames) >= sequence_length:
                    break  # Stop if we have enough frames

            if len(frames) == sequence_length:
                sequence_output_folder = os.path.join(output_folder, action)
                os.makedirs(sequence_output_folder, exist_ok=True)
                output_sequence_folder = os.path.join(sequence_output_folder, sequence)
                os.makedirs(output_sequence_folder, exist_ok=True)
                for i, frame in enumerate(frames):
                    cv2.imwrite(os.path.join(output_sequence_folder, f'frame{i + 1}.jpg'), frame)

# Example usage
if __name__ == "__main__":
    create_image_sequences(
        folder_path=r'D:\AI Projects\building action recognition model\AI-PackageGuard\dataset',
        output_folder=r'D:\AI Projects\building action recognition model\AI-PackageGuard\padded_dataset',
        sequence_length=20
    )
