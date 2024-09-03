# Driver Drowsiness Detection
This project aims to detect drowsiness in drivers using a machine learning model built with OpenCV, 
Keras, and TensorFlow. The system monitors the driver's face, specifically focusing on the eyes, to 
detect signs of drowsiness and alert the driver when necessary. This can help prevent accidents 
caused by drowsiness while driving.

# Features
* Real-time drowsiness detection using webcam input
* Eye aspect ratio calculation to detect eye closure
* Alert system to warn the driver in case of drowsiness
* User-friendly interface for easy deployment
  
# Prerequisites
Before you begin, ensure you have met the following requirements:

* Python 3.7+
* TensorFlow 2.0.0
* Keras
* OpenCV
* NumPy
  
You can install the necessary libraries using the following command:

```Copy code
pip install tensorflow keras opencv-python numpy
```

# Installation
Clone the repository:

```Copy code
git clone https://github.com/prarbdht/Driver-Drowsiness-detection-system.git
```
```
cd Driver_Drowsiness_Detection
```
Install the required Python packages:

```Copy code
pip install -r requirements.txt
```
Download or prepare your dataset (if applicable).

Train the model (if needed) by running:

```Copy code
python model.py
```

# Usage
To run the drowsiness detection system, execute the following command:

```Copy code
python drowsiness_detection.py
```
The script will activate your webcam and start monitoring the driver's face. If drowsiness is detected,
an alert will be triggered.

# Files in the Repository
* drowsiness_detection.py: Main script to run the drowsiness detection system.
* model.py: Script for building and training the machine learning model.
* requirements.txt: List of dependencies required for the project.
* README.md: Documentation file.

# How It Works
The system uses a pre-trained model to monitor the driver's face in real-time. The Eye Aspect Ratio 
(EAR) is computed to detect whether the driver's eyes are closed for an extended period, indicating 
drowsiness. If the EAR falls below a certain threshold, the system triggers an alert.

# Conclusion
Driver drowsiness detection is a critical component in enhancing road safety. This project
demonstrates how machine learning techniques can be applied to create a practical solution that
could potentially save lives by preventing accidents caused by drowsy driving. With further
improvements and optimizations, this system can be integrated into various real-world
applications, contributing to safer driving conditions for everyone.
