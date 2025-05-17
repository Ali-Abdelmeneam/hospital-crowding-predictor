import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Generate synthetic data ===
np.random.seed(42)
crowding_ratios = np.linspace(0.1, 2.0, 500)
labels = (crowding_ratios > 0.9).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "crowding_ratio": crowding_ratios,
    "is_crowded": labels
})

print("\nSample data:")
print(df.head())
print("\nLabel distribution:")
print(df['is_crowded'].value_counts())

# === Train the model ===
X = df[["crowding_ratio"]]
y = df["is_crowded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n✅ Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Save model with metadata ===
joblib.dump({
    "model": model,
    "columns": ["crowding_ratio"]
}, "crowding_model.pkl")

print("\n📦 Model saved as 'crowding_model.pkl'")