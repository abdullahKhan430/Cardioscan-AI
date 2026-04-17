import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MODEL_FILE    = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    # For numerical columns
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    # For categorical columns
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # Lets train the model
    heart = pd.read_csv("heart.csv")

    # Create a stratified split based on target (HeartDisease)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(heart, heart["HeartDisease"]):
        heart.loc[test_index].to_csv("input.csv", index=False)
        heart = heart.loc[train_index]

    heart_labels   = heart["HeartDisease"].copy()
    heart_features = heart.drop("HeartDisease", axis=1)

    num_attribs = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    cat_attribs = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    heart_prepared = pipeline.fit_transform(heart_features)

    # ================= MULTIPLE MODELS =================
    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(random_state=42),
        "Random Forest"       : RandomForestClassifier(random_state=42),
        "SVM"                 : SVC(probability=True, random_state=42)
    }

    print("\n🔄 Training and comparing models...\n")
    print(f"{'Model':<25} {'CV F1 Mean':>12} {'CV F1 Std':>12}")
    print("-" * 52)

    scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, heart_prepared, heart_labels, cv=5, scoring="f1")
        scores[name] = cv_scores.mean()
        print(f"{name:<25} {cv_scores.mean():>12.4f} {cv_scores.std():>12.4f}")

    # ================= BEST MODEL =================
    best_model_name = max(scores, key=scores.get)
    best_model      = models[best_model_name]
    best_model.fit(heart_prepared, heart_labels)

    print(f"\n🏆 Best model: {best_model_name}  (F1 = {scores[best_model_name]:.4f})")

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(pipeline,   PIPELINE_FILE)
    print("Model is trained. Congrats!")

else:
    # Lets do inference
    model    = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")

    # Drop target column if present (it comes from the saved test split)
    input_features = input_data.drop("HeartDisease", axis=1) if "HeartDisease" in input_data.columns else input_data

    transformed_input = pipeline.transform(input_features)
    predictions       = model.predict(transformed_input)
    probabilities     = model.predict_proba(transformed_input)[:, 1]

    # ================= EVALUATION (if actuals available) =================
    if "HeartDisease" in input_data.columns:
        actuals = input_data["HeartDisease"]
        print("\n📊 Model Evaluation on Test Set:")
        print(f"  {'Accuracy':<12}: {accuracy_score(actuals, predictions):.4f}")
        print(f"  {'F1 Score':<12}: {f1_score(actuals, predictions):.4f}")
        print(f"  {'ROC-AUC':<12}: {roc_auc_score(actuals, probabilities):.4f}")

    input_data["prediction"]       = predictions
    input_data["risk_probability"] = probabilities.round(4)
    input_data.to_csv("output.csv", index=False)
    print("\nInference is complete, results saved to output.csv Enjoy!")