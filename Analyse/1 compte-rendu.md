# IZLITNE MAROUANE

<img src="IZLITNE MAROUANE.jpg" width="200" align="left" style="margin-right: 20px; border-radius: 10px;"/>

<br>

**Numéro d'étudiant** : 22006529  
**Classe** : CAC2

<br clear="left"/>

---

# Compte rendu : Analyse Prédictive des Ventes (Business_Sales_EDA)

**Date :** 26 Novembre 2025

---

## À propos du jeu de données

Le jeu de données **Business_Sales_EDA**, utilisé dans cette analyse, recense des transactions de vente détaillées pour divers produits (vêtements, chaussures, vestes). Chaque ligne représente un produit spécifique avec ses caractéristiques intrinsèques et contextuelles.

L'objectif est de prédire la variable cible **Sales Volume** (Volume des ventes) en fonction de divers facteurs marketing et produits tels que :
* **Positionnement** : Emplacement dans le magasin (Aisle, End-cap, Front of Store).
* **Marketing** : Indicateurs de promotion (`Promotion`, `Seasonal`).
* **Caractéristiques Produit** : Catégorie, Prix, Marque, Matériau, Origine.

Ce dataset permet d'évaluer l'impact des stratégies de mise en avant et des caractéristiques produits sur la performance commerciale.

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Analyse Exploratoire des Données (EDA)](#2-analyse-exploratoire-des-données-eda)
    - [2.1 Chargement et Aperçu](#21-chargement-et-aperçu)
    - [2.2 Prétraitement et Encodage](#22-prétraitement-et-encodage)
    - [2.3 Analyse des Valeurs Manquantes](#23-analyse-des-valeurs-manquantes)
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)
    - [3.1 Séparation des Données (Data Split)](#31-séparation-des-données-data-split)
4. [Implémentation des Modèles et Résultats](#4-implémentation-des-modèles-et-résultats)
    - [4.1 Régression Linéaire](#41-régression-linéaire)
    - [4.2 Arbre de Décision (Decision Tree)](#42-arbre-de-décision)
    - [4.3 Forêt Aléatoire (Random Forest)](#43-forêt-aléatoire)
    - [4.4 Support Vector Regressor (SVR)](#44-support-vector-regressor)
    - [4.5 Gradient Boosting Regressor (Le Meilleur Modèle)](#45-gradient-boosting-regressor)
5. [Tableau Comparatif et Analyse](#5-tableau-comparatif-et-analyse)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction et Contexte

L'objectif de ce projet est de développer un modèle de machine learning capable de prédire le **Volume des Ventes** ($Y$) avec la plus grande précision possible. Nous avons comparé plusieurs algorithmes de régression pour déterminer lequel capture le mieux les relations entre les variables explicatives ($X$) et la cible.

La démarche suivie est la suivante :
1.  Nettoyage et encodage des données (traitement des variables catégorielles comme "Promotion" ou "Seasonal").
2.  Séparation des données en ensembles d'entraînement et de test.
3.  Entraînement de 5 modèles distincts.
4.  Comparaison basée sur le $R^2$ (coefficient de détermination), le MAE (erreur absolue moyenne) et le RMSE.

---

## 2. Analyse Exploratoire des Données (EDA)

### 2.1 Chargement et Aperçu

Le dataset est chargé avec Pandas. Nous observons que le fichier utilise le point-virgule (`;`) comme séparateur.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Business_sales_EDA.csv', sep=';') 
print(f"Dimensions du dataset : {df.shape}")
df.head()
```

### 2.2 Prétraitement et Encodage

Les variables telles que `Promotion`, `Seasonal`, et `Product Position` sont de nature **catégorielle**. Nous utilisons le **Label Encoding** pour les transformer en valeurs numériques.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cat_cols = ['Product Position', 'Promotion', 'Product Category', 'Seasonal', 
            'brand', 'section', 'season', 'material', 'origin']

for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

df = df.drop(columns=['url', 'name', 'description', 'currency', 'terms'], errors='ignore')
```

### 2.3 Analyse des Valeurs Manquantes
Avant de modéliser, il est essentiel de vérifier et de gérer les valeurs manquantes (NaN). Nous effectuons une vérification, puis nous procédons à une suppression simple des lignes contenant des valeurs manquantes (df.dropna()) pour garantir l'intégrité des données d'entraînement.
```python
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
df = df.dropna()
```

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation des Données (Data Split)
Nous séparons nos données en deux ensembles pour évaluer la capacité de généralisation des modèles :

Train (80%) : Utilisé pour l'entraînement.

Test (20%) : Utilisé pour l'évaluation finale.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Sales Volume', 'Product ID'])
y = df['Sales Volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Taille de l'ensemble d'entraînement X : {X_train.shape}")
print(f"Taille de l'ensemble de test X : {X_test.shape}")
```

---

## 4. Implémentation des Modèles et Résultats

### 4.1 Régression Linéaire

La Régression Linéaire cherche une relation linéaire directe. Elle sert de modèle de base pour évaluer la performance initiale

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
```
**Résultats :**

  - **$R^2$ ≈ 0.93** (93%)
  - **MAE ≈ 62.39 $**
    
### 4.2 Arbre de Décision

L'Arbre de Décision capture les relations non-linéaires par des divisions conditionnelles successives. Il est rapide mais sujet au sur-apprentissage.

```python
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred_dt))
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
```
**Résultats :**

  - **$R^2$ ≈ 0.87** (87%)
  - **MAE ≈ 82.46 $**
    
### 4.3 Forêt Aléatoire

Le Random Forest utilise un ensemble de nombreux Arbres de Décision et fait la moyenne de leurs prédictions, ce qui réduit la variance et améliore la robustesse.

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
```
**Résultats :**

  - **$R^2$ ≈ 0.93** (93%)
  - **MAE ≈ 63.04 $**

### 4.4 Support Vector Regressor

Le SVR cherche à définir un hyperplan optimal avec une marge d'erreur tolérée. Il est crucial de scaler les données au préalable pour ce modèle, car il est sensible à l'échelle.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)

print("R2 Score:", r2_score(y_test, y_pred_svr))
print("MAE:", mean_absolute_error(y_test, y_pred_svr))
```
**Résultats :**

  - **$R^2$ ≈ 0.63** (63%)
  - **MAE ≈ 137.61 $**
  - **MSE ≈ 32147.57**
  - **RMSE ≈ 179.30**
    
### 4.5 Gradient Boosting Regressor

Le Gradient Boosting construit les arbres séquentiellement. Chaque nouvel arbre est entraîné pour corriger les erreurs résiduelles faites par l'ensemble des arbres précédents, aboutissant souvent à une précision supérieure.

```python
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)

print("R2 Score:", r2_gb)
print("MAE:", mae_gb)
print("RMSE:", rmse_gb)
```
**Résultats :**

  - **$R^2$ ≈ 0.94** (94%)
  - **MAE ≈ 59.08 $**
  - **MSE ≈ 5675.85**
  - **RMSE ≈ 75.34**
    
---

## 5. Tableau Comparatif et Analyse

| Modèle | R² | MAE | MSE | RMSE | Performance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gradient Boosting | 0.94 | 59.08 | 5675.85 | 75.34 | 🏆 Meilleur |
| Régression Linéaire | 0.93 | 62.39 | - | - | ⭐ Très Bon |
| Random Forest | 0.93 | 63.04 | - | - | ⭐ Très Bon |
| Decision Tree | 0.87 | 82.46 | - | - | Moyen |
| SVR | 0.63 | 137.61 | 32147.57 | 179.30 | Faible |

### Analyse des Résultats et Recommandations

1.  **Modèle Optimal** : Le **Gradient Boosting Regressor** est le plus performant, expliquant 94% de la variance des ventes ($R^2=0.94$) avec l'erreur moyenne la plus faible (MAE=59.08).
2.  **Robustesse** : Les modèles basés sur l'ensemble d'arbres (Gradient Boosting et Random Forest) et la Régression Linéaire offrent les meilleurs résultats, suggérant que les données contiennent à la fois des relations linéaires et complexes.
3.  **Prochaines Étapes** : Il est recommandé de procéder à une optimisation fine des hyperparamètres (via GridSearchCV ou RandomizedSearchCV) pour le Gradient Boosting afin de maximiser la performance et d'assurer une meilleure généralisation.

-----

## 6\. Conclusion

Cette analyse prédictive des ventes a démontré l'efficacité des méthodes d'ensemble pour modéliser le volume des ventes. Le modèle **Gradient Boosting Regressor** fournit une base robuste pour la prévision des ventes futures. Ces résultats peuvent directement informer les décisions stratégiques, telles que l'allocation des budgets marketing ou le positionnement des produits, en quantifiant l'impact des différentes caractéristiques sur les revenus.

```
```
