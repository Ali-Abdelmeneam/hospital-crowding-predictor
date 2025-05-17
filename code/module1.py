import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
file_path = r"C:\Users\alime\OneDrive\Desktop\Project AOU 2025\archive\HRR Scorecard_ 20 _ 40 _ 60 - 20 Population.csv"
df = pd.read_csv(file_path)

# Clean and rename columns
columns_to_use = [
    'Total Hospital Beds',
    'Available Hospital Beds',
    'Potentially Available Hospital Beds*',
    'Total ICU Beds',
    'Available ICU Beds',
    'Projected Infected Individuals',
    'Adult Population',
    'Population 65+',
    'Hospital Beds Needed, Six Months'
]
df = df[columns_to_use].copy()
df.columns = [
    'total_hospital_beds', 'available_hospital_beds', 'potentially_available_hospital_beds',
    'total_icu_beds', 'available_icu_beds', 'projected_infected_individuals',
    'adult_population', 'population_65+', 'hospital_beds_needed_six_months'
]

# Convert strings to numeric
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(',', '').str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing and invalid rows
df = df.dropna()
df = df[(df['available_hospital_beds'] > 0) & (df['hospital_beds_needed_six_months'] > 0)]

# Create binary label: Crowded = 1 if hospital beds needed exceeds available
df['is_crowded'] = (df['hospital_beds_needed_six_months'] > df['available_hospital_beds']).astype(int)

print("\nOriginal Class Balance:\n", df['is_crowded'].value_counts())

# Handle class imbalance using SMOTE or class_weight (better than dropping data)
from sklearn.ensemble import RandomForestClassifier

X = df[[
    'total_hospital_beds', 'available_hospital_beds', 'potentially_available_hospital_beds',
    'total_icu_beds', 'available_icu_beds', 'projected_infected_individuals',
    'adult_population', 'population_65+'
]]
y = df['is_crowded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Use class_weight to address imbalance
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'crowding_model.pkl')
print("\n📦 Model saved as 'crowding_model.pkl'")
