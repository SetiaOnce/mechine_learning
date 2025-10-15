print(f'=' * 60)
print('==== LANGKAH 2 : Collection ====')
print(f'=' * 60)
import pandas as pd
df = pd.read_csv("../processed_kelulusan.csv")
print(df.info())
print(df.head())

print(f'=' * 60)
print('==== LANGKAH 3 : Cleaning ====')
print(f'=' * 60)
print(df.isnull().sum())
df = df.drop_duplicates()
import seaborn as sns
sns.boxplot(x=df['IPK'])

print(f'=' * 60)
print('==== LANGKAH 4 : Exploratory Data Analysis (EDA) ====')
print(f'=' * 60)
print(df.describe())
sns.histplot(df['IPK'], bins=10, kde=True)
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")


print(f'=' * 60)
print('==== LANGKAH 5 : Feature Engineering ====')
print(f'=' * 60)
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)

print(f'=' * 60)
print('==== Langkah 6 — Splitting Dataset ====')
print(f'=' * 60)
from sklearn.model_selection import train_test_split

X = df.drop('Lulus', axis=1)
y = df['Lulus']


# KODE SALAH    
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, test_size=0.3, stratify=y, random_state=42)

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# KODE PERBAIKAN
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print(X_train.shape, X_val.shape, X_test.shape)