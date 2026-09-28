# AI Vision and Emotion Recognition System

An AI based computer vision application that performs real time facial emotion recognition, face detection, human detection, and object detection using deep learning.

## Project Overview

This project combines facial emotion recognition with real time computer vision. The system uses a trained deep learning model to classify facial expressions into seven emotion categories.

It also uses YOLO for object detection and OpenCV for face detection and image processing.

The application provides a graphical interface where users can analyze live camera input or upload an image.

## Features

• Seven emotion recognition classes

• Angry

• Disgusted

• Fearful

• Happy

• Neutral

• Sad

• Surprised

• Real time face detection

• Multiple face detection

• Real time emotion recognition

• Emotion confidence score

• Human detection

• Object detection

• Object confidence score

• FPS monitoring

• Emotion smoothing for more stable predictions

• Camera capture

• Image upload and analysis

• Emoji display according to detected emotion

• Detection result logging

• Training accuracy and loss graphs

• Confusion matrix

• Classification report

## Technologies Used

• Python

• TensorFlow

• Keras

• OpenCV

• NumPy

• Pillow

• Scikit Learn

• Matplotlib

• Ultralytics YOLO

• Tkinter

## Project Structure

```text
AI_Vision_Emotion_Analyzer/
│
├── train.py
├── gui.py
├── requirements.txt
│
├── emojis/
│   ├── angry.png
│   ├── disgusted.png
│   ├── fearful.png
│   ├── happy.png
│   ├── neutral.png
│   ├── sad.png
│   └── surprised.png
│
├── data/
│   ├── train/
│   └── test/
│
├── captures/
├── results/
└── training_reports/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI_Vision_Emotion_Analyzer.git
```

Open the project folder:

```bash
cd AI_Vision_Emotion_Analyzer
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Training the Emotion Recognition Model

Place the emotion dataset inside the `data` directory using the required class folders.

Then run:

```bash
python train.py
```

The training process generates:

```text
model.h5
training_reports/
```

The training reports contain the training history, accuracy and loss graphs, confusion matrix, and classification report.

## Running the Application

After training the model, run:

```bash
python gui.py
```

The application opens the AI Vision and Emotion Analyzer interface.

The camera can then be used for real time analysis.

## Emotion Recognition

The system recognizes seven facial emotion categories:

| Emotion   | Description                           |
| --------- | ------------------------------------- |
| Angry     | Detects an angry facial expression    |
| Disgusted | Detects a disgusted facial expression |
| Fearful   | Detects a fearful facial expression   |
| Happy     | Detects a happy facial expression     |
| Neutral   | Detects a neutral facial expression   |
| Sad       | Detects a sad facial expression       |
| Surprised | Detects a surprised facial expression |

## Object Detection

The application uses YOLO for detecting objects in the camera frame.

Detected objects are displayed with:

• Object name

• Bounding box

• Confidence score

The system also counts detected people separately.

## Dataset

The emotion recognition component requires a facial expression dataset organized into seven emotion classes.

The dataset is used only for training and evaluation of the emotion recognition model.

Dataset files are excluded from the GitHub repository using `.gitignore`.

## Applications

This project can be used for:

• Computer vision research

• Human computer interaction

• Emotion aware interfaces

• AI education

• Facial expression analysis

• Real time object detection

• Deep learning demonstrations

## Limitations

Emotion recognition is an AI based prediction and should not be treated as a definitive measurement of a person's actual emotional state.

Performance can vary depending on lighting, camera quality, facial position, image quality, and differences between the training dataset and real world images.

## Future Improvements

Possible future improvements include:

• Improved emotion recognition accuracy

• More advanced CNN architectures

• Transformer based emotion recognition

• Better multi face tracking

• Real time emotion analytics

• Web based interface

• Mobile application

• Cloud based inference

• Larger and more diverse datasets

## License

This project is intended for educational and research purposes.
