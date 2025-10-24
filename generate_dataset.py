import pandas as pd
import numpy as np

# Jumlah data
n = 200
np.random.seed(42)

# Buat kolom data
ipk = np.round(np.random.uniform(2.0, 4.0, n), 2)
absensi = np.random.randint(0, 15, n)
waktu_belajar = np.random.randint(1, 16, n)

# Rasio dan kombinasi fitur
rasio_absensi = absensi / 14
ipk_x_study = np.round(ipk * waktu_belajar, 2)

# Tentukan label Lulus (berdasarkan aturan logis)
lulus = []
for i in range(n):
    score = (ipk[i] * 0.6) + (waktu_belajar[i] / 15 * 0.3) + ((14 - absensi[i]) / 14 * 0.1)
    if score > 2.5:
        lulus.append(1)
    else:
        lulus.append(0)

# Buat DataFrame
df = pd.DataFrame({
    "IPK": ipk,
    "Jumlah_Absensi": absensi,
    "Waktu_Belajar_Jam": waktu_belajar,
    "Lulus": lulus,
    "Rasio_Absensi": rasio_absensi,
    "IPK_x_Study": ipk_x_study
})

# Simpan ke CSV
df.to_csv("processed_kelulusan.csv", index=False)
print("✅ Dataset baru berhasil dibuat: processed_kelulusan.csv")
print(df.head(10))
print(f"\nJumlah data: {len(df)} | Jumlah Lulus: {sum(df['Lulus'])} | Tidak Lulus: {len(df)-sum(df['Lulus'])}")
