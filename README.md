

# 🫀 CardioScan AI — Heart Disease Prediction System

CardioScan AI is a machine learning-based web application that predicts the risk of heart disease using clinical patient data. It combines a trained ML model with an interactive Streamlit UI to provide real-time risk analysis.

---

## 🚀 Live Demo

👉 *Add your deployed link here*
Example: https://your-app-name.streamlit.app

---

## 📌 Features

* 🔍 Predicts heart disease risk using 11 clinical features
* 📊 Displays probability score and risk level (Low / Medium / High)
* 📈 Interactive visualizations (Gauge, Feature Importance, Charts)
* 🧠 Uses multiple ML models and selects the best one
* ⚡ Fast and user-friendly Streamlit interface
* 🎯 Real-time predictions

---

## 🧠 Machine Learning Workflow

1. Dataset loaded from `heart.csv`
2. Data split using Stratified Sampling
3. Preprocessing:

   * Missing value handling (SimpleImputer)
   * Feature scaling (StandardScaler)
   * Encoding categorical variables (OneHotEncoder)
4. Models trained:

   * Logistic Regression
   * Decision Tree
   * Random Forest
   * Support Vector Machine (SVM)
5. Best model selected using **F1 Score**
6. Model and pipeline saved using `joblib`

---

## 🏗️ Project Structure

```
cardioscan-ai/
│
├── app.py              # Streamlit frontend (UI)
├── main.py             # Backend (model training & saving)
├── heart.csv           # Dataset
├── model.pkl           # Trained ML model
├── pipeline.pkl        # Preprocessing pipeline
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/cardioscan-ai.git
cd cardioscan-ai
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Train the model (IMPORTANT)

```
python main.py
```

This will generate:

* `model.pkl`
* `pipeline.pkl`

---

### 4️⃣ Run the app

```
streamlit run app.py
```

---

## 📊 Input Features

The model uses the following 11 features:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Max Heart Rate
* Exercise Angina
* Oldpeak
* ST Slope

---

## 📈 Output

* ✅ Prediction: Heart Disease (Yes / No)
* 📊 Risk Probability (%)
* ❤️ Risk Level (Low / Medium / High)
* 📉 Visual insights & recommendations

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
It is **not a substitute for professional medical advice**.

---

## 👨‍💻 Author

**Abdullah Khan** 🚀

---

## 🌟 Future Improvements

* Add user login system
* Export report as PDF
* Mobile optimization
* Improve model accuracy
* Integrate real-time medical APIs

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
