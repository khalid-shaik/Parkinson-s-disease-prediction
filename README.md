Parkinson’s Disease Prediction Using Speech Data
📘 Overview

Parkinson’s Disease (PD) is a progressive neurological disorder that affects movement, coordination, and speech. It often leads to symptoms such as tremors, muscle stiffness, slowness of movement, difficulty speaking, and mental health challenges.
Although there is no cure, early detection plays a major role in managing symptoms and improving the patient’s quality of life.

This project focuses on detecting Parkinson’s Disease using speech data. Voice changes are one of the early indicators of PD, as the disease affects the muscles used for speech. By analyzing specific vocal features, this model aims to classify whether a person is likely to have Parkinson’s Disease or not.

🧩 Objective

To build a machine learning model that predicts the presence of Parkinson’s Disease using patients’ voice recordings and extracted acoustic features.

🗂️ Dataset

The dataset contains voice measurements from both healthy individuals and Parkinson’s patients.

Each record includes multiple biomedical voice features such as:

Average vocal fundamental frequency (MDVP:Fo(Hz))

Variation in fundamental frequency (MDVP:Fhi(Hz), MDVP:Flo(Hz))

Jitter and Shimmer (frequency and amplitude variations)

Noise-to-Harmonic Ratio (NHR)

Nonlinear measures (RPDE, DFA, etc.)

Source: UCI Machine Learning Repository – Parkinson’s Dataset

⚙️ Technologies Used

Python

NumPy, Pandas – for data preprocessing and analysis

Matplotlib, Seaborn – for data visualization

Scikit-learn – for model building (Support Vector Machine)

Jupyter/Colab Notebook – for implementation

🔬 Methodology

Data Preprocessing:

Handled missing values and normalized features.

Split dataset into training and testing sets (80:20).

Feature Analysis:

Identified key voice parameters contributing to Parkinson’s detection.

Visualized feature correlations using heatmaps.

Model Building:

Trained a Support Vector Machine (SVM) classifier for binary classification (Parkinson’s / Healthy).

Tuned hyperparameters to improve accuracy and stability.

Model Evaluation:

Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrix.

Achieved 91% accuracy on the test set.

📊 Results

The SVM model correctly identified 91 out of 100 cases.

Features such as MDVP:Fo(Hz), Jitter, and Shimmer showed strong influence on classification.

Visual plots helped understand the difference in vocal parameters between healthy and PD subjects.

🚀 Future Improvements

Implement deep learning models (CNN/LSTM) for feature extraction directly from audio signals.

Include larger, real-time voice datasets for higher generalization.

Build an interactive web app for easy voice-based PD screening.
