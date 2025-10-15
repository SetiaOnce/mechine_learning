# NAMA      : I GEDE YOGA SETIAWAN
# NIM       : 231011401028
# KELAS     : 05TPLE016

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(f'=' * 60)
print('==== LANGKAH 1 : Muat Data ====')
print(f'=' * 60)
print(X_train.shape, X_val.shape, X_test.shape)
print()

print(f'=' * 60)
print('==== Langkah 2 — Baseline Model & Pipeline ====')
print(f'=' * 60)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)
print("Baseline (LogReg) F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))
print()

print(f'=' * 60)
print('==== Langkah 3 — Model Alternatif (Random Forest) ====')
print(f'=' * 60)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("RandomForest F1(val):", f1_score(y_val, y_val_rf, average="macro"))
print()

print(f'=' * 60)
print('==== Langkah 4 — Validasi Silang & Tuning Ringkas ====')
print(f'=' * 60)
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# KODE SALAH
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# KODE BENAR
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print("Best RF F1(val):", f1_score(y_val, y_val_best, average="macro"))

print(f'=' * 60)
print('==== Langkah 5 — Evaluasi Akhir (Test Set) ====')
print(f'=' * 60)

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import numpy as np

final_model = best_rf  # atau pipe_lr jika baseline lebih baik
y_test_pred = final_model.predict(X_test)

print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    
    if len(np.unique(y_test)) > 1:  # hanya jalankan jika ada lebih dari 1 kelas
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
        plt.tight_layout()
        plt.savefig("roc_test.png", dpi=120)
    else:
        print("ROC-AUC(test): dilewati (hanya 1 kelas di y_test)")
