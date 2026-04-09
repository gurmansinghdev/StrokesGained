# v1_initial_code.py
# Initial AI-generated code for PGA Tour Strokes Gained Prediction
# Generated using ChatGPT-4 with the prompt:
# "Write Python code to predict strokes gained total for PGA Tour players
#  using linear regression and player statistics."

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("pgaTourData.csv")
# Select features and target
X = df[["Fairway Percentage", "Avg Distance", "gir", "Average Putts", "Average Scrambling"]]
y = df["Average SG Total"]

# Drop missing values
df = df.dropna()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print("R2 Score:", score)
