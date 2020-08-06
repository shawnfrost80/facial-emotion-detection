import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("saved_classifier.model")
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

categories = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

while(video.isOpened()):
    _, frame = video.read()

    faces = face_classifier.detectMultiScale(frame, 1.32, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
        image = cv2.resize(face_img, (48,48))  
        image = np.array(image, dtype=np.float32)
        image /= 255
        image = image.reshape(-1, 48, 48, 1)

        prediction = model.predict([image])
        emotion = categories[np.argmax(prediction[0])]
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()
