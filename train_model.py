import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("housing.csv")

# ---------------------------
# Encode categorical columns
# ---------------------------
binary_cols = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]

for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Encode furnishingstatus
le = LabelEncoder()
df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])

# ---------------------------
# Features & Target
# ---------------------------
X = df.drop('price', axis=1)
y = df['price']

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# ---------------------------
# Coefficients
# ---------------------------
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Importance:")
print(coefficients)

# ---------------------------
# Save Model
# ---------------------------
joblib.dump(model, "house_price_model.pkl")

# ---------------------------
# Example Prediction
# ---------------------------
example_house = [[7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 1]]
prediction = model.predict(example_house)

print("\nExample House Price Prediction:", prediction)
