import cv2
import os
import numpy as np
from keras.models import load_model
from pygame import mixer
import time


# Inicialización de sonido
def initialize_sound():
    mixer.init()
    try:
        return mixer.Sound('alarm.wav')
    except FileNotFoundError:
        raise FileNotFoundError("El archivo 'alarm.wav' no se encontró.")


# Cargar clasificadores en cascada
def load_cascade_classifiers():
    face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
    left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
    return face_cascade, left_eye_cascade, right_eye_cascade


# Cargar el modelo
def load_model_from_file():
    model_path = 'models/cnnCat2.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo '{model_path}' no se encontró en la carpeta 'models/'.")
    return load_model(model_path)


# Preprocesar imagen de ojo para la predicción
def analyze_eye(eye, model):
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (24, 24)) / 255
    eye = eye.reshape(24, 24, -1)
    eye = np.expand_dims(eye, axis=0)
    return np.argmax(model.predict(eye), axis=-1)


# Función principal para la ejecución del programa
def main():
    # Inicialización
    sound = initialize_sound()
    face_cascade, left_eye_cascade, right_eye_cascade = load_cascade_classifiers()
    model = load_model_from_file()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede acceder a la cámara.")
        exit()

    drowsiness_score = 0
    alarm_playing = False
    last_played_time = 0
    sound_duration = 10  # Duración del sonido en segundos

    # Captura de video
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
            right_eye_prediction = analyze_eye(r_eye, model)

        # Análisis del ojo izquierdo
        if len(left_eye) > 0:
            (x, y, w, h) = left_eye[0]
            l_eye = frame[y:y + h, x:x + w]
            left_eye_prediction = analyze_eye(l_eye, model)

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
                except Exception as e:
                    print(f"Error al reproducir sonido: {e}")
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 2)

        # Mostrar el video
        cv2.imshow('Driver Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
