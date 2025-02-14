# Automated-Detection-and-Classification-of-Diabetic-Retinopathy-Using-MobileNetV2-and-DenseNet201

Overview

This project implements an automated system for the detection and classification of diabetic retinopathy using deep learning techniques. The system utilizes MobileNetV2 and DenseNet201 to classify retinal images into five categories:

No Diabetic Retinopathy (No DR)

Mild DR

Moderate DR

Severe DR

Proliferative DR

A Flask-based web application is developed to allow users to upload retinal images and receive classification results.

Features

Deep Learning Models: Uses MobileNetV2 for efficiency and DenseNet201 for high accuracy.

Flask Web Application: Allows users to upload images and select a model for prediction.

Preprocessing Pipeline: Includes dataset preparation, image resizing, normalization, and augmentation.

Model Evaluation: Computes accuracy, precision, recall, F1-score, and confusion matrices.

Visualization: Performance metrics and training progress visualized via charts.

System Requirements

Hardware:

Processor: 11th Gen Intel Core i5-1135G7 @ 2.40 GHz

RAM: 8GB

System Type: 64-bit OS, x64-based processor

Software:

Python 3.x

Flask for web application

TensorFlow & Keras for deep learning

NumPy, Pandas, Matplotlib, Seaborn for data handling and visualization

HTML, CSS, JavaScript for frontend development

Installation

Clone the Repository:

git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

Install Dependencies:

pip install -r requirements.txt

Download Dataset:

Dataset is obtained from Kaggle (Retinopathy dataset with 1,812 images categorized into five classes).

Place dataset in data/ directory.

Run the Flask Application:

python app.py

Access the web application at: http://127.0.0.1:5000

Model Training

Data Preprocessing:

Images resized to 224x224 pixels.

Normalization applied to scale pixel values between [0,1].

Augmentation performed to improve generalization.

Training the Models:

python train.py

MobileNetV2 and DenseNet201 are fine-tuned using transfer learning.

Models are saved as MobileNetV2.h5 and DenseNet201.h5.

Evaluation:

Accuracy, Precision, Recall, and F1-score are computed.

Confusion matrices are visualized to assess model performance.

Using the Web Application

Upload an Image: Users can upload a retinal image for analysis.

Select Model: Choose either MobileNetV2 or DenseNet201.

View Predictions: The application will classify the image and display the results.

Analyze Performance: Charts and performance metrics are available for review.

Project Structure

├── app.py               # Flask web application
├── train.py             # Model training script
├── models/              # Trained models (MobileNetV2.h5, DenseNet201.h5)
├── data/                # Dataset directory
├── static/              # Static files (CSS, JavaScript, Images)
├── templates/           # HTML templates
├── requirements.txt     # Required Python libraries
├── README.md            # Project documentation

Results

MobileNetV2: Achieved high efficiency with lightweight architecture.

DenseNet201: Provided better accuracy but required more computational resources.

Performance Comparison:

MobileNetV2 performed better in real-time classification.

DenseNet201 achieved higher classification accuracy, especially in complex cases.

Future Improvements

Implement Explainable AI (XAI) to interpret predictions.

Optimize models for mobile deployment.

Expand dataset for better generalization.
