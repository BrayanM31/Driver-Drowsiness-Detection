import os
import time
import cv2
from keras.models import load_model
import numpy as np
from pygame import mixer

# Inicialización de sonido
mixer.init()
try:
    sound = mixer.Sound('alarm.wav')
except FileNotFoundError as exc:
    raise FileNotFoundError("El archivo 'alarm.wav' no se encontró.") from exc

# Cargar clasificadores en cascada para detección de rostro y ojos
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Verificar si el modelo existe
if not os.path.exists('models/cnnCat2.h5'):
    raise FileNotFoundError("El modelo 'cnnCat2.h5' no se encontró en la carpeta 'models/'.")

# Cargar el modelo de predicción
eye_detection_model = load_model('models/cnnCat2.h5')

# Configuración inicial de variables
DROWSINESS_SCORE = 0
FRAME_THICKNESS = 2
LAST_PLAYED_TIME = 0
SOUND_DURATION = 10  # Duración del sonido en segundos
ALARM_PLAYING = False

def analyze_eye(eye, model):
    """
    Analiza el ojo para predecir si está cerrado o abierto usando el modelo de red neuronal.
    Args:
        eye: Imagen del ojo.
        model: Modelo de predicción entrenado.
    Returns:
        Predicción de si el ojo está cerrado (0) o abierto (1).
    """
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (24, 24))
    eye = eye / 255
    eye = eye.reshape(24, 24, -1)
    eye = np.expand_dims(eye, axis=0)
    return np.argmax(model.predict(eye), axis=-1)

# Inicialización de variables de predicción para los ojos
right_eye_prediction = [99]
left_eye_prediction = [99]

# Captura de video
cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detección de rostro y ojos
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        left_eye = left_eye_cascade.detectMultiScale(gray)
        right_eye = right_eye_cascade.detectMultiScale(gray)

        # Detección de rostro
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        # Análisis del ojo derecho
        if len(right_eye) > 0:
            (x, y, w, h) = right_eye[0]
            r_eye = frame[y:y + h, x:x + w]
            right_eye_prediction = analyze_eye(r_eye, eye_detection_model)

        # Análisis del ojo izquierdo
        if len(left_eye) > 0:
            (x, y, w, h) = left_eye[0]
            l_eye = frame[y:y + h, x:x + w]
            left_eye_prediction = analyze_eye(l_eye, eye_detection_model)

        # Somnolencia detectada
        if right_eye_prediction[0] == 0 and left_eye_prediction[0] == 0:
            DROWSINESS_SCORE += 1
            cv2.putText(frame, "Closed", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        else:
            DROWSINESS_SCORE = max(DROWSINESS_SCORE - 1, 0)
            cv2.putText(frame, "Open", (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

        # Alarma si somnolencia persiste
        if DROWSINESS_SCORE > 15:
            if not ALARM_PLAYING or time.time() - LAST_PLAYED_TIME >= SOUND_DURATION:
                try:
                    sound.play()
                    ALARM_PLAYING = True
                    LAST_PLAYED_TIME = time.time()
                except Exception as e:
                    print(f"Error al reproducir la alarma: {e}")
                    pass
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), FRAME_THICKNESS)

        # Mostrar el video
        cv2.imshow('Driver Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
