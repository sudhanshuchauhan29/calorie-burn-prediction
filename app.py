# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Load and prepare data
# calories = pd.read_csv("calories.csv")
# exercise = pd.read_csv("exercise.csv")
# data = pd.merge(exercise, calories, on="User_ID")

# if 'Gender' in data.columns:
#     data['Gender'] = data['Gender'].map({'male': 1, 'female': 0})

# X = data.drop(['Calories', 'User_ID'], axis=1)
# y = data['Calories']

# # Train model
# model = LinearRegression()
# model.fit(X, y)

# # --- Streamlit UI ---
# st.title("Calories Predictor 🍎")

# # Dynamically create input fields for each feature
# user_input = {}
# for col in X.columns:
#     if col == "Gender":
#         user_input[col] = st.selectbox("Gender", ["male", "female"])
#     else:
#         user_input[col] = st.number_input(f"Enter {col}", min_value=0.0, value=0.0)

# # Convert input to DataFrame
# input_df = pd.DataFrame([user_input])
# if 'Gender' in input_df.columns:
#     input_df['Gender'] = input_df['Gender'].map({'male': 1, 'female': 0})

# # Predict button
# if st.button("Predict Calories"):
#     prediction = model.predict(input_df)[0]
#     st.success(f"Estimated Calories Burned: {prediction:.2f} kcal")


from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and prepare data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
data = pd.merge(exercise, calories, on="User_ID")

if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'male': 1, 'female': 0})

X = data.drop(['Calories', 'User_ID'], axis=1)
y = data['Calories']

# Train Linear Regression
model = LinearRegression()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Get user inputs
        user_input = {}
        for col in X.columns:
            value = request.form[col]
            if col == "Gender":
                user_input[col] = 1 if value.lower() == "male" else 0
            else:
                user_input[col] = float(value)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Predict
        prediction = model.predict(input_df)[0]
    
    return render_template("index.html", prediction=prediction, columns=X.columns)

if __name__ == "__main__":
    app.run(debug=True)
