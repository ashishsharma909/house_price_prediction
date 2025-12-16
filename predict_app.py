import pandas as pd
import joblib

print("\n🏠 HOUSE PRICE PREDICTION APP 🏠")

# Load trained model
model = joblib.load("house_price_model.pkl")

# Helper function
def yes_no(val):
    return 1 if val == "yes" else 0

# Take user input
area = int(input("Enter area (sq ft): "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
stories = int(input("Enter stories: "))

mainroad = yes_no(input("Main road (yes/no): ").lower())
guestroom = yes_no(input("Guest room (yes/no): ").lower())
basement = yes_no(input("Basement (yes/no): ").lower())
hotwaterheating = yes_no(input("Hot water heating (yes/no): ").lower())
airconditioning = yes_no(input("Air conditioning (yes/no): ").lower())

parking = int(input("Parking spaces: "))
prefarea = yes_no(input("Preferred area (yes/no): ").lower())

furnishing_map = {
    "furnished": 0,
    "semi-furnished": 1,
    "unfurnished": 2
}
furnishingstatus = furnishing_map[
    input("Furnishing (furnished/semi-furnished/unfurnished): ").lower()
]

# Create input dataframe
input_data = pd.DataFrame([{
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwaterheating,
    'airconditioning': airconditioning,
    'parking': parking,
    'prefarea': prefarea,
    'furnishingstatus': furnishingstatus
}])

# Prediction
prediction = model.predict(input_data)[0]

print("\n💰 Predicted House Price: ₹{:,.0f}".format(prediction))
