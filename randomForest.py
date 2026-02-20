# Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

# Initialize and train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
rf_preds = rf.predict(X_test)

# Evaluate model
rf_r2 = r2_score(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print(f"Random Forest Regression → R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")

# Optional: Feature Importance
import matplotlib.pyplot as plt

importances = rf.feature_importances_
plt.barh(X.columns, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()
