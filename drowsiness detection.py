import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Inicialización de sonido
mixer.init()
try:
    sound = mixer.Sound('alarm.wav')
except FileNotFoundError:
    raise FileNotFoundError("El archivo 'alarm.wav' no se encontró.")

# Cargar clasificadores en cascada
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Verificar si el modelo existe
if not os.path.exists('models/cnnCat2.h5'):
    raise FileNotFoundError("El modelo 'cnnCat2.h5' no se encontró en la carpeta 'models/'.")

# Cargar el modelo
model = load_model('models/cnnCat2.h5')

# Configuración inicial
drowsiness_score = 0
frame_thickness = 2
last_played_time = 0
sound_duration = 10  # Duración del sonido en segundos
alarm_playing = False

# Análisis de ojos
def analyze_eye(eye, model):
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (24, 24))
    eye = eye / 255
    eye = eye.reshape(24, 24, -1)
    eye = np.expand_dims(eye, axis=0)
    return np.argmax(model.predict(eye), axis=-1)

# Captura de video
cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detección de rostro y ojos
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        left_eye = left_eye_cascade.detectMultiScale(gray)
        right_eye = right_eye_cascade.detectMultiScale(gray)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            right_eye_prediction = analyze_eye(r_eye, model)
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            left_eye_prediction = analyze_eye(l_eye, model)
            break

        # Somnolencia detectada
        if right_eye_prediction[0] == 0 and left_eye_prediction[0] == 0:
            drowsiness_score += 1
            cv2.putText(frame, "Closed", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        else:
            drowsiness_score = max(drowsiness_score - 1, 0)
            cv2.putText(frame, "Open", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

        # Alarma si somnolencia persiste
        if drowsiness_score > 15:
            if not alarm_playing or time.time() - last_played_time >= sound_duration:
                try:
                    sound.play()
                    alarm_playing = True
                    last_played_time = time.time()
                except:
                    pass
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), frame_thickness)

        # Mostrar el video
        cv2.imshow('Driver Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()