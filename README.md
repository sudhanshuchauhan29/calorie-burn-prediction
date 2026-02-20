🔥 Calorie Burn Prediction Web Application

A Machine Learning-based web application that predicts the number of calories burned during exercise using physiological and workout-related parameters.

This project demonstrates an end-to-end ML pipeline — including data preprocessing, model training, evaluation, comparison of multiple regression algorithms, and deployment using Flask with a responsive HTML/CSS frontend.

📌 Project Overview

The application takes user inputs such as:

Gender

Age

Height

Weight

Duration

Heart Rate

Body Temperature

Based on these parameters, the trained model predicts the estimated calories burned in real time.

🚀 Key Features

Real-time calorie prediction

Comparison of multiple regression models

Flask-based backend integration

Clean and responsive frontend UI

Model serialization using Pickle

Modular project structure

🧠 Machine Learning Models Used

Three regression algorithms were implemented and compared:

Model R² Score RMSE
Linear Regression 0.9673 11.4889
Random Forest Regression 0.9983 2.6526
XGBoost Regression 0.9995 1.4389

👉 XGBoost achieved the best performance and was selected for deployment.

🛠️ Tech Stack
Backend

Python

Flask

Machine Learning

Scikit-learn

XGBoost

Pandas

NumPy

Frontend

HTML5

CSS3

Tools

Pickle (Model Saving)

Git & GitHub

Project Structure
calorie-burn-prediction/
│
├── static/
│ └── style.css
│
├── templates/
│ └── index.html
│
├── LinearRegressor.py
├── randomForest.py
├── XGBoost.py
├── model.pkl
├── app.py
├── requirements.txt
└── README.md

Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/sudhanshuchauhan29/calorie-burn-prediction.git
cd calorie-burn-prediction


2️⃣ Create Virtual Environment
python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt

Or manually:

pip install flask scikit-learn xgboost pandas numpy
4️⃣ Run the Application
python app.py

Open in browser:

http://127.0.0.1:5000/
📊 Model Evaluation Metrics

R² Score (Best Model - XGBoost): 0.9995

RMSE: 1.4389

Compared multiple regression algorithms for performance optimization

🎯 What This Project Demonstrates

End-to-end Machine Learning workflow

Regression modeling and evaluation

Model comparison and optimization

Backend deployment using Flask

Frontend integration with ML model

Clean modular project architecture

👨‍💻 Author

Sudhanshu Chauhan
B.Tech Student
Interested in Machine Learning, Java Backend Development, and Full Stack Development
