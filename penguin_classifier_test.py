import joblib
import pandas as pd

model = joblib.load("penguin_classifier.pkl")

X = pd.DataFrame([{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750
}])

print(model.predict(X))