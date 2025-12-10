# 🏡 Projet Data Science – Prédiction des Prix de l'Immobilier en Californie  
*(Dataset : `fetch_california_housing` – scikit-learn)*[1]

## 1. Contexte métier et mission

### Le problème (Business Case)

En Californie, les acteurs de l'immobilier (agences, promoteurs, banques) doivent estimer rapidement la valeur médiane des logements par bloc géographique pour guider les investissements, fixer les prix de vente et accorder des crédits hypothécaires.[2]
Une sous-évaluation entraîne une perte de revenus, tandis qu'une sur-évaluation expose à des risques de défaut de paiement.[3][2]

**Objectif :** Développer un modèle de régression prédisant la valeur médiane des maisons (en centaines de milliers de dollars) à partir de 8 features socio-démographiques et géographiques.[1]

### L'enjeu métier
- **Décision d'investissement** : Identifier les zones à fort potentiel.  
- **Gestion du risque bancaire** : Évaluation réaliste pour les prêts immobiliers.  
- **Simulation stratégique** : Impact du revenu médian ou de la densité sur les prix.[3]

***

## 2. Les données (l'Input)

**Dataset California Housing** (20 640 échantillons, 8 features numériques continues).[1]

| Élément | Description |  
|---------|-------------|  
| **Samples** | 20 640 blocs de recensement californiens [1] |  
| **Features (X)** | MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude [2] |  
| **Target (y)** | `MedHouseVal` (valeur médiane des maisons ×100k $) [1] |  

***

## 3. Code Python Complet – Cycle de vie

### 3.1 Importation des bibliothèques

```python
# ==============================================================================
# COURS DATA SCIENCE : CYCLE DE VIE COMPLET (SCRIPT PÉDAGOGIQUE)
# PROBLÈME DE RÉGRESSION : PRÉDICTION DES PRIX DES MAISONS EN CALIFORNIE
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
print("✅ Bibliothèques importées.\n")
```

### 3.2 Chargement & simulation données sales

```python
# 2. CHARGEMENT
data = fetch_california_housing(as_frame=True)
df = data.frame
df.rename(columns={'MedHouseVal': 'target'}, inplace=True)
print(f"📊 Dataset : {df.shape}")

# 3. SIMULATION DONNÉES SALES (5% NaN)
np.random.seed(42)
features_columns = df.columns[:-1]
df_dirty = df.copy()
for col in features_columns:
    df_dirty.loc[df_dirty.sample(frac=0.05, random_state=42).index, col] = np.nan
print(f"🕳️  NaN générés : {df_dirty.isnull().sum().sum()}\n")
```

### 3.3 Nettoyage (Data Wrangling + Scaling)

```python
# 4. NETTOYAGE
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

# A. Imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)
print("✅ Imputation OK")

# B. Scaling (CRUCIAL pour régression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_clean_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns)
print("✅ Scaling OK\n")
```

**💡 Expert** : En prod, split → fit(imputer/scaler) sur Train → transform Train/Test.[4]

***

## 4. Analyse Exploratoire (EDA)

```python
# 5. EDA
print("📈 EDA...")

# Stats cible
print("Statistiques target :\n", y.describe())

# Histogramme cible
plt.figure(figsize=(10, 5))
sns.histplot(y, kde=True, bins=50)
plt.title("Distribution Prix Maisons ($100k)")
plt.show()

# Corrélations avec target
plt.figure(figsize=(10, 8))
corr_matrix = pd.concat([X_clean, y], axis=1).corr()
sns.heatmap(corr_matrix[['target']].sort_values('target', ascending=False),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Corrélations avec Prix")
plt.show()
```

**Insights** : `MedInc` ≈ +0.7 corrélation. Distribution skewed (prix plafonnés).[2]

***

## 5. Split Train/Test

```python
# 6. SPLIT 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_clean_scaled, y, test_size=0.2, random_state=42
)
print(f"🚂 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")
```

**random_state=42** = reproductibilité scientifique.[4]

***

## 6. Modélisation : RandomForestRegressor 🌲

```python
# 7. MODÈLE
print("🤖 Entraînement Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Modèle entraîné\n")
```

**Pourquoi RF ?** Réduit variance (bagging + feature randomness). Robust to outliers.[5]

***

## 7. Évaluation (Métriques Régression)

```python
# 8. ÉVALUATION
y_pred = model.predict(X_test)

# Métriques
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"🎯 R² : {r2:.4f}")
print(f"📏 RMSE: {rmse:.4f} ($100k)")

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Réalité ($100k)"); plt.ylabel("Prédiction ($100k)")
plt.title('Réalité vs Prédiction')
plt.show()

print("\n🏁 FIN")
```

**Interprétation** : RMSE=0.5 ≈ erreur moyenne 50k$. R²>0.8 = excellent pour baseline.[2]

***

## 8. Pipeline Complet – Schéma

```
📥 Données Sales → 🧹 Imputation → ⚖️ Scaling → 🔀 Split → 🌲 RF → 📊 Métriques
   (5% NaN)       → (mean)      → (StdScaler) → 80/20  → (100 trees) → R²/RMSE
```

***

## 9. Récapitulatif Technique

| Phase | Outil | Résultat Attendu |
|-------|-------|------------------|
| **Chargement** | `fetch_california_housing()` | (20640, 9) [1] |
| **Nettoyage** | `SimpleImputer(mean)` | 0 NaN |
| **Scaling** | `StandardScaler()` | μ=0, σ=1 par feature |
| **Modèle** | `RandomForestRegressor(100)` | R² ≈ 0.80-0.85 [5] |
| **Éval** | RMSE en $100k | <0.55 typique |

***

## 10. Conclusion Projet

Ce pipeline complet transforme un **problème métier** (évaluation immobilière) en **solution IA actionable** : de l'acquisition à l'évaluation, en passant par un EDA métier et un modèle robuste.[5][2]
**Prochaines étapes** : Feature engineering (interactions géo/revenu), GridSearchCV, déploiement Streamlit.

[1](https://sklearn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
[2](https://www.classes.cs.uchicago.edu/archive/2021/fall/12100-1/pa/pa5/dataset-houseprice.html)
[3](https://irays-teknology-ltd.com/BLOG/California-Housing/)
[4](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html)
[5](https://dataloop.ai/library/model/rajistics_california_housing/)
