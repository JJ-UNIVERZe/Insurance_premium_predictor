# ✈️ Travel Safety & Insurance Premium Predictor

An interactive **Streamlit web app** that predicts **travel insurance risk** and **premium costs** based on a traveler’s personal and trip details.  
Built with **machine learning models**, it provides instant premium estimates, explainability.

---

## 🚀 Features

- **Risk Probability Estimation**  
  Predict the likelihood of filing a claim during your trip.

- **Expected Claim Amount Prediction**  
  Estimate the average claim amount if a claim occurs.

- **Premium Calculation Engine**  
  Calculates an estimated premium using actuarial-style formulas based on predicted risk and claim amount.

- **Explainability (SHAP)**  
  Visualize which features influenced the model’s predictions.

---

## 🗂 Project Structure
```text
INSURANCE-APP/
├── .venv/
├── data/
│   ├── insurance_dataset.csv
│   └── travel_augmented.csv
├── models/
│   ├── claim_pipeline.joblib
│   ├── preprocessor.joblib
│   └── risk_pipeline.joblib
├── src/
│   ├── app/
│   │   ├── EDA.ipynb
│   │   ├── explain.py
│   │   └── premium.py
│   ├── data/
│   │   └── augment.py
│   ├── features/
│   │   └── preprocessor.py
│   └── models/
│       └── train_models.py
├── app.py
├── README.md
└── requirements.txt
```

---

## 🛠 Technologies Used

- Streamlit
- Scikit-learn — machine learning models  
- Pandas — data processing  
- SHAP — explainability  
---

## 📦 Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/JJ-UNIVERZE/INSURANCE-APP.git
   cd INSURANCE-APP
   ```

2.Create and activate virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3.Install dependencies
 ```bash
     pip install -r requirements.txt
 ```
4.Download Dataset From
 ```bash
    https://www.kaggle.com/datasets/ravalsmit/insurance-claims-and-policy-data
 ```
5.Run the app locally
 ```bash
   streamlit run app.py
 ```

  
