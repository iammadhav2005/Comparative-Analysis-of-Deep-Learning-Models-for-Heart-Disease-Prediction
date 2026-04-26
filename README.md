Heart Disease Prediction using Machine Learning and Deep Learning
Overview

This project focuses on predicting the risk of heart disease using machine learning and deep learning models. It performs a comparative analysis of multiple models to determine the most effective approach for structured clinical data. The system also includes a graphical user interface (GUI) for real-time prediction based on patient input.

Objectives
Develop an accurate model for early heart disease detection
Analyze key medical features such as age, cholesterol, and blood pressure
Compare multiple models including MLP, CNN, RNN, and SVM
Identify the best-performing model based on evaluation metrics
Build a user-friendly system for real-time prediction
Dataset
Source: UCI Machine Learning Repository
Dataset: Heart Disease Dataset (Cleveland)
Samples: 303 patient records
Features: Age, sex, chest pain type, cholesterol, blood pressure, ECG results, etc.
Target: Presence or absence of heart disease
Technologies Used
Python
TensorFlow and Keras
Scikit-learn
Pandas and NumPy
Matplotlib
Tkinter (for GUI)
Project Workflow
Data Collection
Data Preprocessing (handling missing values, encoding, normalization)
Train-Test Split
Model Development (MLP, CNN, RNN, SVM)
Model Training and Evaluation
Performance Comparison
GUI Development for real-time prediction
Models Implemented
Multi-Layer Perceptron (MLP)
Convolutional Neural Network (CNN)
Recurrent Neural Network (RNN)
Support Vector Machine (SVM)
Results
SVM (RBF Kernel): Highest accuracy (~81%)
MLP: Strong performance (~77%)
CNN: Moderate performance (~71%)
RNN: Lower performance (~69%)

The results show that traditional machine learning models like SVM can outperform deep learning models when working with structured tabular data.

Features
Comparative analysis of multiple models
Data preprocessing and feature scaling
Visualization of training performance
GUI for easy user interaction
Real-time prediction with confidence score
Applications
Clinical decision support systems
Early heart disease detection
Preventive healthcare screening
Telemedicine and remote monitoring
Medical research and analysis
Limitations
Dataset size is relatively small
Performance may vary with different datasets
Some models require further hyperparameter tuning
Future Work
Use larger and more diverse datasets
Improve model accuracy through optimization
Deploy as a web or mobile application
Integrate explainable AI features
Conclusion

This project demonstrates the effectiveness of machine learning and deep learning techniques in predicting heart disease. It highlights the importance of selecting the right model based on data type and shows that simpler models can sometimes outperform more complex architectures.

References
UCI Machine Learning Repository – Heart Disease Dataset
Research papers on heart disease prediction using ML and DL models
