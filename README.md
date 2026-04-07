# Loan Eligibility Prediction Model

**Course:** CST2216 — Business Intelligence System Infrastructure  
**College:** Algonquin College  
**Project:** Modularizing and Deploying ML Code (Week 10)

---

## Project Overview

This project modularises a Jupyter Notebook–based classification pipeline into a production-ready Python package and deploys it as an interactive Streamlit web application. The model predicts whether a bank loan applicant should be approved based on 13 demographic and financial attributes drawn from the German Credit dataset (614 records).

Three classifiers are trained and compared:
- Logistic Regression
- Decision Tree
- Random Forest (with hyperparameter tuning)

**Success criterion:** accuracy ≥ 76% on the held-out test set.

---

## Project Structure

```
loan-eligibility/
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   └── credit.csv          # Raw dataset (add manually — not committed to Git)
├── models/                 # Saved .pkl model files (auto-created by pipeline)
├── logs/                   # Log files (auto-created at runtime)
└── src/
    ├── __init__.py
    ├── data_loader.py      # CSV loading and validation
    ├── preprocessor.py     # Imputation, encoding, scaling
    ├── model.py            # Training, evaluation, cross-validation
    ├── utils.py            # Logging setup, chart helpers, model persistence
    └── pipeline.py         # End-to-end orchestration script
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/loan-eligibility.git
cd loan-eligibility
```

### 2. Create and activate a virtual environment

```bash
py -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 4. Add the dataset

Place `credit.csv` inside the `data/` folder:

```
data/credit.csv
```

The dataset must contain these columns:

| Column | Type | Description |
|---|---|---|
| Loan_ID | string | Applicant identifier |
| Gender | categorical | Male / Female |
| Married | categorical | Yes / No |
| Dependents | categorical | 0 / 1 / 2 / 3+ |
| Education | categorical | Graduate / Not Graduate |
| Self_Employed | categorical | Yes / No |
| ApplicantIncome | numeric | Monthly income ($) |
| CoapplicantIncome | numeric | Co-applicant income ($) |
| LoanAmount | numeric | Loan amount in $000s |
| Loan_Amount_Term | numeric | Term in months |
| Credit_History | binary | 1 = has history, 0 = none |
| Property_Area | categorical | Urban / Semiurban / Rural |
| Loan_Approved | categorical | Y / N (target) |

---

## Running the App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Tabs

| Tab | What it does |
|---|---|
| 📊 Data Overview | Load dataset, inspect distributions, missing values |
| 🤖 Train & Evaluate | Train all three models, view accuracy, confusion matrices, CV scores and feature importances |
| 🔮 Predict | Enter applicant details and get a live loan eligibility prediction |

---

## Running the Pipeline from the Command Line

```bash
py src/pipeline.py
```

This runs the full pipeline, prints a summary table and saves trained models to `models/`.

---

## Logging

All runtime events are written to `logs/loan_eligibility.log` and echoed to the console. Log level is `INFO` by default.

---

## Live Demo

Deployed on Streamlit Cloud: **[INSERT LINK AFTER DEPLOYMENT]**

---

## Dependencies

| Package | Version |
|---|---|
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| scikit-learn | 1.5.0 |
| matplotlib | 3.9.0 |
| seaborn | 0.13.2 |
| streamlit | 1.35.0 |
| joblib | 1.4.2 |
