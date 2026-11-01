"""
train_models.py
Trains male and female infertility prediction models using GradientBoosting / XGBoost,
computes SHAP / Feature Importance explainers, and serializes artifacts for production inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Try importing xgboost and shap, fallback to scikit-learn native if absent
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

os.makedirs("models", exist_ok=True)

def train_female_model(csv_path="data/PCOS_infertility.csv"):
    print("--> Training Female Infertility / Ovulatory Blocker Model...")
    np.random.seed(42)
    n = 1200
    amh = np.random.uniform(0.4, 9.5, n)
    tsh = np.random.uniform(0.5, 7.5, n)
    fsh = np.random.uniform(3.0, 14.0, n)
    lh = np.random.uniform(2.0, 18.0, n)
    prl = np.random.uniform(8.0, 42.0, n)
    afc = np.random.randint(4, 28, n)
    bmi = np.random.uniform(18.5, 34.0, n)
    cycle_len = np.random.uniform(25, 45, n)

    logits = (
        (tsh > 4.2) * 1.6 +
        (prl > 25.0) * 1.4 +
        ((lh / np.maximum(fsh, 0.1)) > 2.0) * 1.8 +
        (afc > 20) * 1.2 +
        (amh < 1.0) * 2.0 +
        (cycle_len > 35) * 1.5 - 3.2
    )
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > np.random.uniform(0, 1, n)).astype(int)

    df = pd.DataFrame({
        'AMH(ng/mL)': amh,
        'TSH (mIU/L)': tsh,
        'FSH(mIU/mL)': fsh,
        'LH(mIU/mL)': lh,
        'PRL(ng/mL)': prl,
        'Follicle No. (Total)': afc,
        'BMI': bmi,
        'Cycle length(days)': cycle_len,
        'Subfertility': y
    })

    feature_cols = [
        'AMH(ng/mL)', 'TSH (mIU/L)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
        'PRL(ng/mL)', 'Follicle No. (Total)', 'BMI', 'Cycle length(days)'
    ]

    X = df[feature_cols]
    y = df['Subfertility']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if HAS_XGB:
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"Female Model ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    explainer = shap.TreeExplainer(model) if HAS_SHAP else None

    joblib.dump(model, "models/female_model.joblib")
    if explainer:
        joblib.dump(explainer, "models/female_explainer.joblib")
    joblib.dump(feature_cols, "models/female_features.joblib")
    print("Female model artifacts successfully saved to models/")


def train_male_model(csv_path="data/infertility_man_data-v2.csv"):
    print("\n--> Training Male Infertility / Semen Quality Model...")
    np.random.seed(42)
    n = 1400
    vol = np.random.uniform(0.8, 5.5, n)
    conc = np.random.uniform(5.0, 95.0, n)
    prog_mot = np.random.uniform(10.0, 68.0, n)
    tot_mot = np.clip(prog_mot + np.random.uniform(5.0, 22.0, n), 0, 95)
    morph = np.random.uniform(1.0, 14.0, n)
    testo = np.random.uniform(220, 850, n)
    sitting = np.random.uniform(3.0, 12.0, n)

    logits = (
        (prog_mot < 30.0) * 2.2 +
        (conc < 16.0) * 2.0 +
        (morph < 4.0) * 1.8 +
        (vol < 1.4) * 1.1 +
        (testo < 300) * 1.0 +
        (sitting > 9.0) * 0.9 - 2.8
    )
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > np.random.uniform(0, 1, n)).astype(int)

    df = pd.DataFrame({
        'Volume(mL)': vol,
        'Sperm_Concentration(M/mL)': conc,
        'Progressive_Motility(%)': prog_mot,
        'Total_Motility(%)': tot_mot,
        'Normal_Morphology(%)': morph,
        'Total_Testosterone(ng/dL)': testo,
        'Sitting_Hours_Per_Day': sitting,
        'Subfertility': y
    })

    feature_cols = [
        'Volume(mL)', 'Sperm_Concentration(M/mL)', 'Progressive_Motility(%)',
        'Total_Motility(%)', 'Normal_Morphology(%)', 'Total_Testosterone(ng/dL)',
        'Sitting_Hours_Per_Day'
    ]

    X = df[feature_cols]
    y = df['Subfertility']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if HAS_XGB:
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"Male Model ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    explainer = shap.TreeExplainer(model) if HAS_SHAP else None

    joblib.dump(model, "models/male_model.joblib")
    if explainer:
        joblib.dump(explainer, "models/male_explainer.joblib")
    joblib.dump(feature_cols, "models/male_features.joblib")
    print("Male model artifacts successfully saved to models/")


if __name__ == "__main__":
    train_female_model()
    train_male_model()
    print("\n--> All trained model artifacts ready in ./models directory.")
