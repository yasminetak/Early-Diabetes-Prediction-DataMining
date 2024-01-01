# Early-Diabetes-Prediction-DataMining

## Overview 
This project focuses on predicting the likelihood of diabetes in individuals of Pima Indian heritage based on diagnostic measurements. The dataset used is the Pima Indians Diabetes dataset, originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

## Dataset Description

https://www.kaggle.com/datasets/mathchi/diabetes-data-set
The dataset comprises the following features:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration 2 hours after an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skinfold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Binary class variable (0 or 1) indicating the presence or absence of diabetes

## Data Vizualisation
Data_viz.ipynb

This Jupyter Notebook is dedicated to the visualization of the Diabetes dataset. It provides a detailed exploration of the data, presenting key insights through various visualizations such as histograms, scatter plots, and correlation matrices. The goal is to gain a better understanding of the dataset's structure and identify potential patterns or trends.

## Data Preprocessing
data-viz-process.py

The data preprocessing file focuses on preparing the dataset for subsequent analysis. It includes steps such as handling missing values, feature scaling, and encoding categorical variables. The objective is to create a clean and standardized dataset that can be effectively utilized for predictive modeling.

## Data Mining Techniques
Data_Mining_Diabetes.ipynb

The project utilizes a variety of data mining techniques for early diabetes prediction:

K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Random Forest
Decision Tree
Naive Bayes
Neural Network Architectures:
    Convolutional Neural Network (CNN)
    Recurrent Neural Network (RNN)
    Multi-Layer Perceptron (MLP) Classifier

## Application 
app.py 

This file contains the Flask application for an interactive web interface. Run this file to launch the web application and explore the predictions visually. 
To run the Flask application, follow these steps:

1. Ensure you have Flask installed. If not, install it using:

    ```bash
    pip install Flask
    ```

2. Navigate to the project directory in the terminal:

    ```bash
    cd diabetes-prediction
    ```

3. Run the Flask application:

    ```bash
    python app.py
    ```

4. Open a web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the interactive web interface.


## Usage 
1. Clone the repository
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction

2. Install requirements
pip install -r requirements.txt

