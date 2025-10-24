# ==========================================================
# NAMA  : I GEDE YOGA SETIAWAN
# NIM   : 231011401028
# KELAS : 05TPLE016
# ==========================================================
# 🧠 Machine Learning: Prediksi Kelulusan Mahasiswa
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
import joblib

# ==========================================================
# LANGKAH 1 — MUAT DATA
# ==========================================================
print("="*60)
print("==== LANGKAH 1 : MUAT DATA ====")
print("="*60)

df = pd.read_csv("processed_kelulusan.csv")
print(f"Jumlah data: {len(df)}")
print(f"Fitur: {list(df.columns)}")
print(df.head(), "\n")

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split data: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Data shape:")
print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)
print("Distribusi label (train):")
print(y_train.value_counts(normalize=True), "\n")


# ==========================================================
# LANGKAH 2 — PIPELINE & BASELINE RANDOM FOREST
# ==========================================================
print("="*60)
print("==== LANGKAH 2 — PIPELINE & BASELINE RANDOM FOREST ====")
print("="*60)

num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
])

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", rf)])
pipe.fit(X_train, y_train)

y_val_pred = pipe.predict(X_val)
print("Baseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3), "\n")


# ==========================================================
# LANGKAH 3 — VALIDASI SILANG (CROSS-VALIDATION)
# ==========================================================
print("="*60)
print("==== LANGKAH 3 — VALIDASI SILANG ====")
print("="*60)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print(f"CV F1-macro (train): {scores.mean():.4f} ± {scores.std():.4f}\n")


# ==========================================================
# LANGKAH 4 — GRID SEARCH (TUNING PARAMETER)
# ==========================================================
print("="*60)
print("==== LANGKAH 4 — GRID SEARCH (TUNING PARAMETER) ====")
print("="*60)

param = {
  "clf__max_depth": [None, 10, 20, 30],
  "clf__min_samples_split": [2, 5, 10],
  "clf__n_estimators": [200, 300, 400]
}

gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
best_model = gs.best_estimator_

y_val_best = best_model.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))
print(classification_report(y_val, y_val_best, digits=3), "\n")


# ==========================================================
# LANGKAH 5 — EVALUASI AKHIR (TEST SET)
# ==========================================================
print("="*60)
print("==== LANGKAH 5 — EVALUASI AKHIR (TEST SET) ====")
print("="*60)

final_model = best_model
y_test_pred = final_model.predict(X_test)
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):\n", confusion_matrix(y_test, y_test_pred, labels=[0,1]))

# ROC-AUC
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_test_proba)
        print(f"ROC-AUC(test): {auc:.4f}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test Set)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_test.png", dpi=120)

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
        plt.figure(figsize=(5,4))
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Test Set)")
        plt.tight_layout()
        plt.savefig("pr_test.png", dpi=120)
    else:
        print("ROC-AUC(test) dilewati (hanya ada 1 kelas di y_test)")


# ==========================================================
# LANGKAH 6 — PENTINGNYA FITUR
# ==========================================================
print("="*60)
print("==== LANGKAH 6 — PENTINGNYA FITUR ====")
print("="*60)

try:
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out()
    feat_imp = pd.DataFrame({"Fitur": fn, "Importance": importances})
    feat_imp = feat_imp.sort_values("Importance", ascending=False)
    print(feat_imp.head(10), "\n")

    # Visualisasi
    plt.figure(figsize=(7,4))
    plt.barh(feat_imp["Fitur"][:10], feat_imp["Importance"][:10])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
except Exception as e:
    print("Feature importance tidak tersedia:", e)


# ==========================================================
# LANGKAH 7 — SIMPAN MODEL
# ==========================================================
print("="*60)
print("==== LANGKAH 7 — SIMPAN MODEL ====")
print("="*60)
joblib.dump(final_model, "rf_model.pkl")
print("Model disimpan sebagai rf_model.pkl\n")


# ==========================================================
# LANGKAH 8 — CEK INFERENCE (UJI COBA INPUT MANUAL)
# ==========================================================
print("="*60)
print("==== LANGKAH 8 — CEK INFERENCE (UJI COBA INPUT MANUAL) ====")
print("="*60)

mdl = joblib.load("rf_model.pkl")
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4*7
}])
pred = int(mdl.predict(sample)[0])
print("Prediksi Kelulusan:", "Lulus ✅" if pred == 1 else "Tidak Lulus ❌")
