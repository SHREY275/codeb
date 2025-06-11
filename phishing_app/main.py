import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utilitis.explain import shap_explain, lime_explain

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

app = FastAPI()

# ✅ 20 selected features
feature_names = [
    'page_rank', 'google_index', 'nb_www', 'nb_hyperlinks', 'phish_hints',
    'nb_slash', 'web_traffic', 'domain_age', 'shortest_word_path',
    'nb_dots', 'nb_hyphens', 'length_hostname', 'length_url',
    'ratio_digits_url', 'domain_registration_length', 'links_in_tags',
    'domain_in_brand', 'longest_word_path', 'length_words_raw',
    'ratio_extHyperlinks'
]

# Load model and scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


class PhishingInput(BaseModel):
    features: list[float]

class URLRequest(BaseModel):
    url: str


@app.post("/predict")
def predict(input_data: PhishingInput):
    try:
        features = np.array(input_data.features).reshape(1, -1)

        # Check for feature length mismatch
        if features.shape[1] != len(feature_names):
            raise HTTPException(status_code=400, detail=f"Expected {len(feature_names)} features, got {features.shape[1]}")

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        label = "Phishing" if prediction[0] == 1 else "Legitimate"
        return {"prediction": label}
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/shap")
def explain_shap(data: PhishingInput):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    shap_values = shap_explain(model, X_scaled, X_scaled)
    explanation = {
        feature_names[i]: float(shap_values[0].values[i])
        for i in range(len(feature_names))
    }
    return {"shap_explanation": explanation}


@app.post("/explain/lime")
def explain_lime(data: PhishingInput):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    lime_result = lime_explain(model, X_scaled, scaler.transform(np.array([data.features])), feature_names)
    return {"lime_explanation": lime_result}


# ------------------------------
# ✅ Training block
# ------------------------------
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("dataset_phishing.csv")

    # Use only the selected 20 features
    X = df[feature_names]
    y = df['status']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, preds):.2f}")
    print(f"Precision: {precision_score(y_test, preds):.2f}")
    print(f"Recall:    {recall_score(y_test, preds):.2f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    auc = roc_auc_score(y_test, probs)
    print(f"ROC AUC Score: {auc:.2f}")

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')

    # SHAP Explanation
    explainer_shap = shap.Explainer(model, X_train_scaled)
    shap_values = explainer_shap(X_test_scaled[:100])
    shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_names)

    # LIME Explanation
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=["Legitimate", "Phishing"],
        mode='classification'
    )
    exp = explainer_lime.explain_instance(X_test_scaled[0], model.predict_proba)
    exp.save_to_file('lime_explanation.html')
