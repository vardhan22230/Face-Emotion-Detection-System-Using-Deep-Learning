# Face-Emotion-Detection-System-Using-Deep-Learning
  
# Emotion Detection Using CNN  

This project implements a **real-time emotion detection system** using **Convolutional Neural Networks (CNN)** with the **Keras** deep learning framework and **OpenCV** for face detection. The model is trained on grayscale facial images and can classify emotions into seven categories:  
**Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.**  

## 📂 Project Structure  

- **`main.py`** - Trains the CNN model using facial expression dataset.  
- **`test.py`** - Detects emotions in real-time from webcam input using the trained model.  
- **`testdata.py`** - Tests the trained model on a single image.  
- **`haarcascade_frontalface_default.xml`** - Pre-trained face detection model from OpenCV.  

## 🚀 Features  

✔️ **CNN-Based Model** - Uses multiple convolutional and pooling layers to extract features.  
✔️ **Real-Time Face Detection** - Uses OpenCV’s Haar Cascade classifier to detect faces.  
✔️ **Data Augmentation** - Uses Keras' `ImageDataGenerator` to improve model generalization.  
✔️ **Emotion Classification** - Predicts emotions from facial expressions in images/videos.  

## 📌 Installation  

1️⃣ Clone the repository:  
```sh
git clone https://github.com/yourusername/emotion-detection.git  
cd emotion-detection
```
2️⃣ Install required dependencies:  
```sh
pip install -r requirements.txt
```
3️⃣ Run the training script:  
```sh
python main.py
```
4️⃣ Run the real-time emotion detection:  
```sh
python test.py
```
5️⃣ Test on a single image:  
```sh
python testdata.py
```

## 🖼️ Model Training  

The CNN model consists of **4 convolutional layers**, each followed by **ReLU activation** and **MaxPooling**, with **Dropout layers** to prevent overfitting. The final **softmax layer** predicts one of the seven emotion classes.  

## 🛠 Dependencies  

- Python 3.x  
- TensorFlow / Keras  
- OpenCV  
- NumPy  

## 🎯 Future Improvements  

- Train on a larger dataset for better accuracy.  
- Deploy the model as a web or mobile application.  
- Implement real-time emotion analysis with graphical insights.  

📌 **Author:** Y Saivardhan  
