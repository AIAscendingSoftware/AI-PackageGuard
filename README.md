# Custom Action Recognition Model (CNN + LSTM)

This repository contains the code for building a custom action recognition model using CNN-LSTM.

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/custom-action-recognition.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model
1. Extract frames from videos:
   ```bash
   python scripts/extract_frames.py
   ```
2. Preprocess the frames:
   ```bash
   python scripts/preprocess_data.py
   ```
3. Train the model:
   ```bash
   python scripts/train.py
   ```

### Prediction
To predict actions on a new video:
```bash
python scripts/predict.py --video_path="data/raw_videos/test_video.mp4"
