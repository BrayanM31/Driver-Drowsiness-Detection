import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

# Variables para controlar el sonido y el temporizador
alarm_playing = False
last_played_time = 0  # Para controlar cuando fue la última vez que se reprodujo el sonido
sound_duration = 10  # Duración del sonido en segundos

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if rpred[0] == 1:
            lbl = 'Open' 
        if rpred[0] == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if lpred[0] == 1:
            lbl = 'Open'   
        if lpred[0] == 0:
            lbl = 'Closed'
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    if score < 0:
        score = 0   

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    if score > 15:
        # Persona está somnolienta, se activa la alarma
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        
        # Mostrar el mensaje en la terminal
        print("¡Alerta! El usuario está somnoliento. La alarma se activa.")

        # Reproducir el sonido solo si no se está reproduciendo o si pasó suficiente tiempo desde la última vez
        current_time = time.time()  # Tiempo actual en segundos
        if not alarm_playing or current_time - last_played_time >= sound_duration:
            try:
                sound.play()  # Iniciar el sonido
                alarm_playing = True  # Marcar que el sonido está sonando
                last_played_time = current_time  # Actualizar el tiempo de la última vez que se reprodujo
            except:
                pass

        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    else:
        # Si el puntaje cae por debajo del umbral, asegurarse de que el sonido no vuelva a sonar
        if alarm_playing and not mixer.get_busy():
            alarm_playing = False  # Resetear el estado de la alarma

    # Mostrar el frame
    cv2.imshow('frame', frame)

    # Si se presiona 'q', se detiene el ciclo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
