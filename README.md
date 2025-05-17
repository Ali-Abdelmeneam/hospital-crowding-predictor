# 🏥 Hospital Crowding Predictor

This is a graduation project built to help hospitals forecast crowding in critical sections like ICU, Emergency, and General Ward using machine learning.

---

## 📌 Overview

This system uses a trained **Random Forest Classifier** model that predicts whether a hospital section will be **crowded** based on a custom-calculated **crowding ratio**.

The web interface is built using **Streamlit**, and predictions are powered by real and synthetic data to improve accuracy and generalization.

---

## ⚙️ Features

- Interactive **Streamlit web app**
- Crowd prediction for different hospital sections
- Data inputs: Bed availability, infected patients, demographics
- Dynamic charts and trends
- PDF and CSV report generation
- Clean, minimal UI for hospitals with limited infrastructure

---

## 🧠 How it Works

1. The user inputs hospital data (bed counts, patients, population).
2. The system calculates the `crowding_ratio`:
crowding_ratio = projected_infected_individuals / (available_beds + potentially_available_beds)

yaml
Copy
Edit
3. This ratio is passed into a trained machine learning model.
4. The model returns a prediction: **Crowded** or **Not Crowded**.
5. Charts and reports are generated for hospital planning.

---

## 🛠️ Tech Stack

| Layer            | Tech                          |
|------------------|-------------------------------|
| Frontend         | Streamlit                     |
| Backend Model    | Python + scikit-learn (RandomForest) |
| Visualization    | Matplotlib, Altair            |
| PDF Export       | fpdf                           |
| Dataset Source   | [Kaggle Hospital Beds Dataset](https://www.kaggle.com/) |
| Tools Used       | Visual Studio Code, GitHub    |

---

## 📁 Project Structure

📦 Project Folder
│
├── app.py # Streamlit app
├── retrain_crowding_model.py # Model training script
├── crowding_model.pkl # Trained ML model
├── README.md # Project description
└── test.py # Testing script

yaml
Copy
Edit

---

## 🧪 Model Performance

- Accuracy: **100%** on balanced test set (after handling class imbalance)
- Confusion Matrix: Perfect predictions on sample test set

---

## 🧾 Author

**Ali Abdelmeneam**  
Student ID: 21510563  
Graduation Year: 2025

---

## 📜 License

This project is for educational and demonstration purposes only.
