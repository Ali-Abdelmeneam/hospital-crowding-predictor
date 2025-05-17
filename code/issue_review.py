import joblib
import pandas as pd

# === Load model ===
model_data = joblib.load("crowding_model.pkl")
model = model_data["model"]
expected_columns = model_data["columns"]

# === Test input where crowding ratio is just above the threshold ===
test_ratio = 1.4  # Try values like 0.85, 0.91, 1.2 etc.
test_input = pd.DataFrame([[test_ratio]], columns=expected_columns)

print("\n🔍 Testing with Crowding Ratio:", test_ratio)
print("Model Input:")
print(test_input)

# === Predict ===
prediction = model.predict(test_input)[0]
print("\n🧠 Model Output:", prediction)

if prediction == 1:
    print("✅ This section is CROWDED")
else:
    print("❌ This section is NOT crowded")