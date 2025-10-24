# ==========================================================
# NAMA  : I GEDE YOGA SETIAWAN
# NIM   : 231011401028
# KELAS : 05TPLE016
# ==========================================================
# 📘 Machine Learning — Prediksi Kelulusan Mahasiswa
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

# ==========================================================
# LANGKAH 1 — MUAT DATA
# ==========================================================
print("="*60)
print("==== LANGKAH 1 : MUAT DATA ====")
print("="*60)

df = pd.read_csv("processed_kelulusan.csv")
print("Jumlah data:", len(df))
print("Kolom:", list(df.columns))
print("\n5 Data Pertama:")
print(df.head(), "\n")

# Informasi tambahan dataset
print("Ringkasan Statistik:")
print(df.describe(), "\n")

print("Distribusi Target (Lulus):")
print(df["Lulus"].value_counts(normalize=True).rename("Proporsi (%)") * 100)
print()

# Heatmap korelasi fitur
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Korelasi Antar Fitur")
plt.tight_layout()
plt.savefig("heatmap_korelasi.png", dpi=120)

# Split data
X = df.drop("Lulus", axis=1)
y = df["Lulus"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print("Shape Data:")
print("Train:", X_train.shape, "| Validation:", X_val.shape, "| Test:", X_test.shape)
print("="*60, "\n")


# ==========================================================
# LANGKAH 2 — BASELINE MODEL (LOGISTIC REGRESSION)
# ==========================================================
print("==== LANGKAH 2 — BASELINE MODEL (LOGISTIC REGRESSION) ====")
print("="*60)

num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
])

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])
pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)

print("Baseline (LogReg) — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))
print("="*60, "\n")


# ==========================================================
# LANGKAH 3 — MODEL ALTERNATIF (RANDOM FOREST)
# ==========================================================
print("==== LANGKAH 3 — MODEL ALTERNATIF (RANDOM FOREST) ====")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])
pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)

print("RandomForest — F1(val):", f1_score(y_val, y_val_rf, average="macro"))
print(classification_report(y_val, y_val_rf, digits=3))
print("="*60, "\n")


# ==========================================================
# LANGKAH 4 — VALIDASI SILANG & TUNING
# ==========================================================
print("==== LANGKAH 4 — VALIDASI SILANG & TUNING RINGKAS ====")
print("="*60)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param = {
  "clf__max_depth": [None, 10, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("Best Params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)
best_rf = gs.best_estimator_

y_val_best = best_rf.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))
print(classification_report(y_val, y_val_best, digits=3))
print("="*60, "\n")


# ==========================================================
# LANGKAH 5 — EVALUASI AKHIR (TEST SET)
# ==========================================================
print("==== LANGKAH 5 — EVALUASI AKHIR (TEST SET) ====")
print("="*60)

final_model = best_rf
y_test_pred = final_model.predict(X_test)
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (test):\n", cm)

# Visualisasi Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=120)

# ROC-AUC dan Precision-Recall Curve
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_test_proba)
        print(f"ROC-AUC(test): {auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test Set)"); plt.legend()
        plt.tight_layout(); plt.savefig("roc_test.png", dpi=120)

        prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Test Set)")
        plt.tight_layout(); plt.savefig("pr_test.png", dpi=120)
    else:
        print("ROC-AUC dilewati (kelas tidak seimbang).")
print("="*60, "\n")


# ==========================================================
# LANGKAH 6 — PENTINGNYA FITUR
# ==========================================================
print("==== LANGKAH 6 — PENTINGNYA FITUR ====")
print("="*60)

try:
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out()
    feat_imp = pd.DataFrame({"Fitur": fn, "Importance": importances})
    feat_imp = feat_imp.sort_values("Importance", ascending=False)
    print(feat_imp.head(10), "\n")

    plt.figure(figsize=(7,4))
    # sns.barplot(x="Importance", y="Fitur", data=feat_imp.head(10), palette="viridis")
    sns.barplot(
        x="Importance",
        y="Fitur",
        data=feat_imp.head(10),
        palette="viridis",
        hue="Importance",
        legend=False
    )
    plt.title("Top 10 Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
except Exception as e:
    print("Feature importance tidak tersedia:", e)
print("="*60, "\n")
