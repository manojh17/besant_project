import joblib
import numpy as np

model = joblib.load("fertilizer_model.pkl")
encoders = joblib.load("label_encoders.pkl")
soil_encoder = encoders['Soil Type']
crop_encoder = encoders['Crop Type']
fertilizer_encoder = encoders['Fertilizer Name']


soil_options = list(soil_encoder.classes_)
crop_options = list(crop_encoder.classes_)


temp = int(input(" enter temperature? "))
humidity = int(input("enter humidity level "))
moisture = int(input("enter soil moisture level "))

print("\n Choose soil type:")
print(", ".join(soil_options))
soil_type = input("Soil Type: ").strip()
while soil_type not in soil_options:
    print("Case sensitive alert.")
    soil_type = input("Soil Type: ").strip()

print("\n Choose crop type:")
print(", ".join(crop_options))
crop_type = input("Crop Type: ").strip()
while crop_type not in crop_options:
    print("only enter the crop type in the options with exact cases")
    crop_type = input("Crop Type: ").strip()

nitrogen = int(input("enter nitrogen "))
potassium = int(input("enter potassium level: "))
phosphorous = int(input("enter phosphorous level: "))

soil_val = soil_encoder.transform([soil_type])[0]
crop_val = crop_encoder.transform([crop_type])[0]
data = np.array([[temp, humidity, moisture, soil_val, crop_val, nitrogen, potassium, phosphorous]])

#prediction
predicted = model.predict(data)[0]
fertilizer = fertilizer_encoder.inverse_transform([predicted])[0]

print("\n recommanded fertilizer for your field", fertilizer)
