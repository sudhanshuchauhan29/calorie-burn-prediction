# Import Libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load datasets
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge on User_ID
data = pd.merge(exercise, calories, on="User_ID")

# Encode Gender
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'male': 1, 'female': 0})

# Separate features and target
X = data.drop(['Calories', 'User_ID'], axis=1)
y = data['Calories']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize XGBoost Regressor
xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

# Predict
xgb_preds = xgb.predict(X_test)

# Evaluate
xgb_r2 = r2_score(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))

print(f"XGBoost Regression → R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.4f}")