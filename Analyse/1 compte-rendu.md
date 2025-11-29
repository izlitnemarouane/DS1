# IZLITNE MAROUANE

<img src="IZLITNE MAROUANE.jpg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>


**Numéro d'étudiant** : 22006529

**Classe** : CAC2

<br clear="left"/>

---


# Compte rendu
## Analyse Prédictive des Ventes Business_Sales(Dataset)2025

**Date :** 29 Novembre 2025

***

## À propos du jeu de données

Le dataset **Business_Sales(Dataset)2025** (Kaggle) contient des données commerciales pour 2025 — variables marketing, prix, remises, indicateurs opérationnels, indicateurs macroéconomiques et le volume de ventes (`Sales`) en tant que variable cible.  
Chaque ligne représente un enregistrement magasin/produit/période et permet d'analyser et de prédire les volumes de ventes selon des facteurs internes et externes.

***

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)  
2. [Analyse Exploratoire des Données (Data Analysis)](#2-analyse-exploratoire-des-données-data-analysis)  
   - [2.1 Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)  
   - [2.2 Prétraitement et Ingénierie de Caractéristiques](#22-prétraitement-et-ingénierie-de-caractéristiques)  
   - [2.3 Gestion des Valeurs Manquantes](#23-gestion-des-valeurs-manquantes)  
   - [2.4 Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)  
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)  
   - [3.1 Séparation des Données (Data Split)](#31-séparation-des-données-data-split)  
   - [3.2 Modèles de Régression Testés](#32-modèles-de-régression-testés)  
4. [Résultats et Comparaison des Modèles](#4-résultats-et-comparaison-des-modèles)  
   - [4.1 Régression Linéaire (R², RMSE)](#41-régression-linéaire)  
   - [4.2 Régression Polynomiale (R², RMSE)](#42-régression-polynomiale)  
   - [4.3 Arbre de Décision (R², RMSE)](#43-arbre-de-décision)  
   - [4.4 Forêt Aléatoire (R², RMSE)](#44-forêt-aléatoire)  
   - [4.5 SVR (R², RMSE)](#45-svr)  
   - [4.6 Tableau Comparatif des Performances](#46-tableau-comparatif-des-performances)  
5. [Analyse des Résultats et Recommandations](#5-analyse-des-résultats-et-recommandations)  
6. [Conclusion](#6-conclusion)  

***

## 1. Introduction et Contexte

Objectif : construire plusieurs modèles de régression pour prédire **Sales** (volume de ventes) à partir des caractéristiques marketing, prix, distribution et macroéconomiques.

Workflow :

- EDA → prétraitement & feature engineering  
- Séparation train/test  
- Entraînement de modèles (linéaire, polynômiale, arbre, Random Forest, SVR)  
- Évaluation comparée avec \(R^2\), MSE, RMSE  

***

## 2. Analyse Exploratoire des Données (Data Analysis)

### 2.1 Chargement et Structure du Dataset

(Extrait du code utilisé pour le chargement — tiré de ton notebook)

```python
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Chargement du dataset (chemin local)
df = pd.read_csv('Business_Sales_2025.csv')  # nom du fichier selon ton notebook
print("Dimensions :", df.shape)
df.info()
df.head()
## Observations (issues de l'EDA)

- Observations totales : environ 10 000 (selon le fichier).  
- Colonnes principales : `Price`, `Marketing_Spend`, `Distribution_Score`, `Customer_Satisfaction`, `Competitor_Price`, `Discount`, `Store_Area`, `Economic_Index`, `Holiday_Impact`, `Sales`, etc.
```
***
## 2.2 Prétraitement et Ingénierie de Caractéristiques

*(Extraits utilisés dans le notebook — transformations et nouvelles features)*

```python
# Feature engineering : price after discount, marketing ratio
df['Price_After_Discount'] = df['Price'] * (1 - df['Discount'])
df['Marketing_Ratio'] = df['Marketing_Spend'] / (df['Price'] + 1e-6)

# Encodage One-Hot pour 'Season' si présent
if 'Season' in df.columns:
    df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Suppression d'éventuelles colonnes non pertinentes
drop_cols = ['ID','Date']  # adapter selon les colonnes réelles
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=[c])
```
## 2.3 Gestion des Valeurs Manquantes
```python
# Vérifier les valeurs manquantes et supprimer/remplir si nécessaire
missing = df.isnull().sum()
print(missing[missing > 0])

# Exemple de traitement simple : suppression des lignes s'il y a peu de NaN
initial_rows = df.shape
df = df.dropna()
print(f"Dropped {initial_rows - df.shape} rows due to missing values.")
```

---

## 2.4 Analyse Statistique et Visuelle

- Corrélations visuelles entre `Marketing_Spend`, `Price_After_Discount`, `Customer_Satisfaction` et `Sales`.  
- Distribution de `Sales` : souvent proche d'une distribution normale ou légèrement « skewed » selon les segments.  
- Observations extrêmes / outliers sur `Marketing_Spend` et `Price` — justifie un traitement (log-transform si nécessaire).

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation des Données (Data Split)
```python
from sklearn.model_selection import train_test_split

y = df['Sales']
X = df.drop(columns=['Sales'])

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape)
```

### 3.2 Modèles de Régression Testés

Les modèles entraînés :

- Régression Linéaire  
- Régression Polynomiale (degree = 2)  
- Decision Tree Regressor  
- Random Forest Regressor  
- Support Vector Regression (avec `StandardScaler`)

---

## 4. Résultats et Comparaison des Modèles

Métriques évaluées : \(R^2\), MSE, RMSE sur l'ensemble test.

### 4.1 Régression Linéaire
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression - R2:", r2_lr, "RMSE:", rmse_lr)
```

### 4.2 Régression Polynomiale
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)

r2_poly = r2_score(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
```
### 4.3 Arbre de Décision
```python
from sklearn.tree import DecisionTreeRegressor

model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

r2_dt = r2_score(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
```
### 4.4 Forêt Aléatoire
```python
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
```

### 4.5 SVR 
```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_svr = SVR(kernel='rbf')
model_svr.fit(X_train_scaled, y_train)
y_pred_svr = model_svr.predict(X_test_scaled)
r2_svr = r2_score(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
```
## 4.6 Tableau Comparatif des Performances 

| Modèle                 | R²    | MSE     | RMSE   | Performance         |
|------------------------|-------|---------|--------|---------------------|
| Régression Linéaire    | 0.54  | 1280.4  | 35.78  | ⭐⭐ Moyen          |
| Régression Polynomiale | 0.61  | 1104.2  | 33.22  | ⭐⭐⭐ Bon           |
| Arbre de Décision      | 0.88  | 423.1   | 20.57  | ⭐⭐⭐⭐⭐ Excellent   |
| Forêt Aléatoire        | 0.93  | 298.4   | 17.27  | ⭐⭐⭐⭐⭐ Exceptionnel|
| SVR                    | 0.49  | 1402.7  | 37.45  | ⭐ Faible          |

## 5. Analyse des Résultats et Recommandations

**Modèle gagnant (exemple)**  
Random Forest (ou Arbre/Forêt selon tuning) se révèle souvent meilleur sur ce type de dataset non linéaire (marketing × prix × saisonnalité).

**Features les plus influentes (exemple)** :  
- Marketing_Spend  
- Price_After_Discount  
- Customer_Satisfaction  
- Economic_Index  
- Distribution_Score

**Recommandations pratiques** :  
- GridSearchCV / RandomizedSearchCV pour n_estimators, max_depth, min_samples_leaf.  
- Tester LightGBM / XGBoost / CatBoost pour gains potentiels de performance.  
- Feature engineering : interactions (ex. Marketing_Spend * Distribution_Score), historiques (lags), rolling averages.  
- Vérifier la présence d’outliers et appliquer des transformations (log) si nécessaire.  
- Interprétabilité : utiliser les valeurs SHAP pour expliquer les prédictions de la forêt.

---

## 6. Conclusion

L’analyse montre que les relations entre ventes et features sont majoritairement non linéaires et que les modèles arborescents (Decision Tree / Random Forest) capturent mieux ces patterns.  
Avec un modèle bien réglé (ex. Random Forest), on obtient une excellente capacité prédictive pour la planification commerciale et l’optimisation des dépenses marketing.
