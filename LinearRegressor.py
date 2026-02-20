from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

# Load and merge datasets
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge on 'User_ID'
data = pd.merge(exercise, calories, on="User_ID")

# Encode Gender if exists
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'male': 1, 'female': 0})

# Drop unnecessary columns
X = data.drop(['Calories', 'User_ID'], axis=1)
y = data['Calories']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"Linear Regression → R²: {r2:.4f}, RMSE: {rmse:.4f}")
