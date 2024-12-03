from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D

# Modelo secuencial para clasificación
model = Sequential([
    # Primera capa de convolución
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),

    # Segunda capa de convolución
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),

    # Tercera capa de convolución
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),

    # Dropout para evitar sobreajuste
    Dropout(0.25),

    # Aplanar la salida para la capa densa
    Flatten(),

    # Capa densa intermedia
    Dense(128, activation='relu'),

    # Otro Dropout
    Dropout(0.5),

    # Capa de salida con dos clases (ojo abierto/cerrado)
    Dense(2, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Información adicional
print("Modelo configurado y compilado correctamente.")
