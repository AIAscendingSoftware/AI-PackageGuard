import os
import shutil

class FrameSequenceProcessor:
    def __init__(self, target_length=20):
        self.target_length = target_length  # Target number of frames per sequence
    
    def pad_sequence(self, frames, last_frame_name, sequence_folder):
        """If sequence is shorter than target length, pad by copying the last frame with sequential numbering."""
        padded_frames = list(frames)  # Copy the original frames
        last_frame_num = int(last_frame_name.split('.')[0].split('_')[-1])  # Get the last frame number
        last_frame_ext = last_frame_name.split('.')[-1]  # Get the file extension
        
        last_frame_path = os.path.join(sequence_folder, last_frame_name)

        # Check if the last frame exists
        if not os.path.exists(last_frame_path):
            raise FileNotFoundError(f"Last frame '{last_frame_path}' not found!")

        # Add copies of the last frame with incrementing names until we reach the target length
        while len(padded_frames) < self.target_length:
            last_frame_num += 1
            new_frame_name = f"frame_{last_frame_num:04d}.{last_frame_ext}"  # Keep original extension
            padded_frames.append(new_frame_name)
            new_frame_path = os.path.join(sequence_folder, new_frame_name)

            # Copy the last frame to the new frame path
            shutil.copy(last_frame_path, new_frame_path)  

        return padded_frames

    def load_sequence_frames(self, sequence_folder):
        """Load all frames from a sequence folder."""
        frames = []
        
        # Load all frames in sorted order
        for frame_name in sorted(os.listdir(sequence_folder)):
            frame_path = os.path.join(sequence_folder, frame_name)
            if os.path.isfile(frame_path):
                frames.append(frame_name)  # Store frame names

        return frames

    def process_and_save_padded_sequences(self, data_folder, action_labels, output_folder):
        """Process all sequences for each action class, pad if necessary, and save with sequential naming."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # Create output folder if it doesn't exist
        
        for action_label in action_labels:
            action_folder = os.path.join(data_folder, action_label)
            action_output_folder = os.path.join(output_folder, action_label)
            
            if not os.path.exists(action_output_folder):
                os.makedirs(action_output_folder)  # Create action folder in output

            for sequence_folder in os.listdir(action_folder):
                sequence_path = os.path.join(action_folder, sequence_folder)
                sequence_output_folder = os.path.join(action_output_folder, sequence_folder)
                
                if not os.path.exists(sequence_output_folder):
                    os.makedirs(sequence_output_folder)  # Create sequence folder in output

                frames = self.load_sequence_frames(sequence_path)

                # Only pad if the sequence is shorter than the target length
                if len(frames) < self.target_length:
                    frames = self.pad_sequence(frames, frames[-1], sequence_path)  # Pass the last frame name for padding

                # Save the original + padded frames into the new sequence folder
                for frame_name in frames:
                    output_frame_path = os.path.join(sequence_output_folder, frame_name)
                    
                    # Copy the frame (original or duplicated with new name)
                    original_frame_path = os.path.join(sequence_path, frame_name)
                    shutil.copy(original_frame_path, output_frame_path)

        print("Padded sequences with sequential names have been saved successfully!")

# Example usage
if __name__ == '__main__':
    processor = FrameSequenceProcessor(target_length=20)
    
    data_folder = r'D:\AI Projects\building action recognition model\AI-PackageGuard\dataset'  # Base folder containing action sequences
    action_labels = ['throwing_parcels', 'handling_parcels_properly', 'sitting_on_parcels']
    
    # Define an output folder for the padded sequences
    output_folder = r'D:\AI Projects\building action recognition model\AI-PackageGuard\padded_dataset'
    
    processor.process_and_save_padded_sequences(data_folder, action_labels, output_folder)
