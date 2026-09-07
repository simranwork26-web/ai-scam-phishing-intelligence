import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SPLIT_DIR = "data/processed/phiusiil_domain_splits"
MODEL_DIR = "results/phiusiil"

FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

# label 0 = phishing, label 1 = legitimate
# Convert target so positive class = phishing.
def evaluate(model, X, y, name):
    phishing_y = (y == 0).astype(int)
    phishing_prob = 1.0 - model.predict_proba(X)[:, 1]
    phishing_pred = (model.predict(X) == 0).astype(int)

    print(f"\n=== {name} ===")
    print("Accuracy:", f"{accuracy_score(phishing_y, phishing_pred):.4f}")
    print("Precision:", f"{precision_score(phishing_y, phishing_pred):.4f}")
    print("Recall:", f"{recall_score(phishing_y, phishing_pred):.4f}")
    print("F1:", f"{f1_score(phishing_y, phishing_pred):.4f}")
    print("ROC-AUC:", f"{roc_auc_score(phishing_y, phishing_prob):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(phishing_y, phishing_pred))

train = pd.read_csv(f"{SPLIT_DIR}/train.csv")
val = pd.read_csv(f"{SPLIT_DIR}/validation.csv")
test = pd.read_csv(f"{SPLIT_DIR}/test.csv")

X_train = train[FEATURES]
y_train = train["label"]

X_val = val[FEATURES]
y_val = val["label"]

X_test = test[FEATURES]
y_test = test["label"]

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print("Training PhiUSIIL URL-only Logistic Regression...")
model.fit(X_train, y_train)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, f"{MODEL_DIR}/phiusiil_url_logreg.joblib")

evaluate(model, X_val, y_val, "VALIDATION")
evaluate(model, X_test, y_test, "TEST")
