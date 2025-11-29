# ============================================
# 0. Import des librairies
# ============================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 1. Chargement des données
# ============================================
df = pd.read_csv("Business_sales_EDA.csv", sep=";")

print("========= Résumé du Dataset =========")
print(f"Dimensions : {df.shape}")
df.info()
print("\n========= Premières lignes =========")
print(df.head())

# ============================================
# 2. Prétraitement
# ============================================

# 2.1 Suppression de colonnes non pertinentes (adapter la liste)
cols_to_drop = ["url", "description", "terms", "name", "currency"]
df = df.drop(columns=cols_to_drop, errors="ignore")
print("\nColonnes supprimées (si présentes) :", cols_to_drop)

# 2.2 Encodage des variables catégorielles (adapter les noms)
categorical_cols = [
    "section", "brand", "season", "material", "origin",
    "Product Position", "Promotion", "Seasonal"
]

categorical_cols = [c for c in categorical_cols if c in df.columns]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("Variables catégorielles encodées.")
print("Nouvelles dimensions :", df.shape)

# 2.3 Vérification des valeurs manquantes
print("\n========= Valeurs manquantes =========")
print(df.isnull().sum())

# Option simple : suppression des lignes avec NaN
df = df.dropna()
print("Dimensions après dropna :", df.shape)

# ============================================
# 3. Analyse exploratoire rapide
# ============================================

# 3.1 Distribution de la variable cible (adapter le nom de la cible)
target_col = "Sales"  # ou "Sales Volume" selon ton CSV

plt.figure(figsize=(10, 4))
sns.histplot(df[target_col], kde=True)
plt.title("Distribution du volume de ventes")
plt.show()

# 3.2 Matrice de corrélation
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm")
plt.title("Matrice de corrélation (Business sales)")
plt.show()

# ============================================
# 4. Séparation des données
# ============================================
y = df[target_col]
X = df.drop(columns=[target_col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTaille X_train :", X_train.shape)
print("Taille X_test  :", X_test.shape)

# ============================================
# 5. Modèles de régression
# ============================================

results = []

# 5.1 Régression Linéaire
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

results.append(["Régression Linéaire", r2_lr, rmse_lr])

print("\n=== Régression Linéaire ===")
print(f"R²   : {r2_lr:.4f}")
print(f"MSE  : {mse_lr:.2f}")
print(f"RMSE : {rmse_lr:.2f}")

# 5.2 Arbre de Décision
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

results.append(["Arbre de Décision", r2_dt, rmse_dt])

print("\n=== Arbre de Décision ===")
print(f"R²   : {r2_dt:.4f}")
print(f"MSE  : {mse_dt:.2f}")
print(f"RMSE : {rmse_dt:.2f}")

# 5.3 Forêt Aléatoire
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

results.append(["Forêt Aléatoire", r2_rf, rmse_rf])

print("\n=== Forêt Aléatoire ===")
print(f"R²   : {r2_rf:.4f}")
print(f"MSE  : {mse_rf:.2f}")
print(f"RMSE : {rmse_rf:.2f}")

# ============================================
# 6. Tableau comparatif
# ============================================
df_results = pd.DataFrame(results, columns=["Modèle", "R2", "RMSE"])
print("\n========= Résultats comparatifs =========")
print(df_results)

# ============================================
# 7. Visualisation des performances
# ============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.barplot(x="Modèle", y="R2", data=df_results)
plt.title("Comparaison des R²")
plt.xticks(rotation=15)

plt.subplot(1, 2, 2)
sns.barplot(x="Modèle", y="RMSE", data=df_results)
plt.title("Comparaison des RMSE")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

