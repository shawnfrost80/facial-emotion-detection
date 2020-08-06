# Realtime Facial Emotion Detection using Tensorflow and OpenCV
This is Deep Neural Network model (CNN)
-It will detect your emotion as "angry", "disgust", "fear", "happy", "neutral", "sad" and "surprise" based on your facial expression.
-I trained it for 300 epochs with accuracy of "0.94" and validation accuracy of "0.57".

# Dependencies Required

1. Tensorflow - pip install tensorflow
2. OpenCV - pip intall opencv-python
3. Numpy - pip install numpy
4. Dataset - Download it from here https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset, and extract it in the same folder.

# How to run

## For Testers:

- Directly run the "detector.py" file with python for Realtime Facial Emotion Detection.

## For Developers:

- Run the "generate_data.ipynb" file. It will convert the images from the datasets into array, split them into training and testing data and store them as "X_train", "X_test", "y_train", "y_test".
- Then run the "classifier.ipynb" file. This will initialize the CNN model and train it with the data created by "generate_data.ipynb" file.
- Now run the "detector.py" file for Realtime Facial Emotion Detection.

