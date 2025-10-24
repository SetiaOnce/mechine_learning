# ==========================================================
# NAMA  : I GEDE YOGA SETIAWAN
# NIM   : 231011401028
# KELAS : 05TPLE016
# ==========================================================

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os

# ==========================================================
print("=" * 60)
print("==== Langkah 1 — Siapkan Data ====")
print("=" * 60)

df = pd.read_csv("processed_kelulusan.csv")

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

sc = StandardScaler()
Xs = sc.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(X_train.shape, X_val.shape, X_test.shape)

# ==========================================================
# Konfigurasi Eksperimen
# ==========================================================
EXPERIMENTS = [
    {"dense": 32, "optimizer": "adam", "dropout": 0.3, "reg": None},
    {"dense": 64, "optimizer": "adam", "dropout": 0.3, "reg": None},
    {"dense": 128, "optimizer": "adam", "dropout": 0.5, "reg": "l2"},
    {"dense": 64, "optimizer": "sgd", "dropout": 0.3, "reg": "l2"},
]

results = []
os.makedirs("results", exist_ok=True)

# ==========================================================
# Fungsi membangun model dinamis
# ==========================================================
def build_model(input_dim, dense_units, optimizer, dropout=0.3, reg=None):
    reg_obj = None
    if reg == "l2":
        reg_obj = regularizers.l2(0.001)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(dense_units, activation="relu", kernel_regularizer=reg_obj),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    if optimizer == "adam":
        opt = keras.optimizers.Adam(1e-3)
    elif optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    else:
        raise ValueError("Optimizer tidak dikenal")

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    return model


# ==========================================================
# Jalankan semua eksperimen
# ==========================================================
for exp in EXPERIMENTS:
    print("=" * 60)
    print(f"Mulai Eksperimen: {exp}")
    print("=" * 60)

    model = build_model(X_train.shape[1], exp["dense"], exp["optimizer"], exp["dropout"], exp["reg"])

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    # Evaluasi model
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    # Simpan hasil
    results.append({
        "dense": exp["dense"],
        "optimizer": exp["optimizer"],
        "dropout": exp["dropout"],
        "reg": exp["reg"],
        "acc": acc,
        "f1": f1,
        "auc": roc,
        "loss": loss,
    })

    # Plot Learning Curve
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve (dense={exp['dense']}, opt={exp['optimizer']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/curve_dense{exp['dense']}_{exp['optimizer']}.png", dpi=120)
    plt.close()

# ==========================================================
# Simpan hasil ke CSV
# ==========================================================
df_results = pd.DataFrame(results)
df_results.to_csv("results/experiment_results.csv", index=False)
print("\n✅ Semua eksperimen selesai. Hasil disimpan di 'results/experiment_results.csv'")
print(df_results)
