
# IZLITNE MAROUANE

<img src="https://image2url.com/images/1765362786985-df3bb0b1-e113-40f7-a0cc-80d894c711cb.jpg"
     alt="Logo marouane izlitne"
     style="height:300px; margin-right:300px; float:left; border-radius:10px;">

<br><br clear="left"/>



**Numéro d'étudiant** : 22006529  
**Classe** : CAC2

# Vidéo de présentation : https://drive.google.com/file/d/1jYMS5vVjKU5KlbxXeOxNK072qVyZ89WM/view?usp=sharing

# 🏡 Projet Data Science – Prédiction des Prix de l'Immobilier en Californie  
*(Dataset : `fetch_california_housing` – scikit-learn)*[1]

## Table des Matières

- [1. Contexte métier et mission](#1-contexte-métier-et-mission)
- [Les données (l'Input)](#les-données-linput)
- [2. Le Code Python (Laboratoire)](#2-le-code-python-laboratoire)
- [3. Analyse Approfondie : Nettoyage (Data Wrangling)](#3-analyse-approfondie--nettoyage-data-wrangling)
- [4. Analyse Approfondie : Exploration (EDA)](#4-analyse-approfondie--exploration-eda)
- [5. Analyse Approfondie : Méthodologie (Split)](#5-analyse-approfondie--méthodologie-split)
- [6. FOCUS THÉORIQUE : L’Algorithme Random Forest](#6-focus-théorique--lalgorithme-random-forest-)
- [7. Analyse Approfondie : Les Métriques de Régression](#7-analyse-approfondie-)
- [8. Pipeline Complet – Schéma](#8-pipeline-complet--schéma)
- [9. Récapitulatif Technique](#9-récapitulatif-technique)
- [Conclusion du Projet](#conclusion-du-projet)

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

##  Les données (l'Input)

**Dataset California Housing** (20 640 échantillons, 8 features numériques continues).[1]

| Élément | Description |  
|---------|-------------|  
| **Samples** | 20 640 blocs de recensement californiens [1] |  
| **Features (X)** | MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude [2] |  
| **Target (y)** | `MedHouseVal` (valeur médiane des maisons ×100k $) [1] |  

***

## 2. Le Code Python (Laboratoire)
```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# --- PHASE 1 : ACQUISITION & SIMULATION ---
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()   # Features + target déjà combinés
df.rename(columns={'MedHouseVal': 'target'}, inplace=True)

print(df.head())
print(df.shape)

# Simulation de la réalité (Données sales)
np.random.seed(42)
df_dirty = df.copy()

# On corrompt 5% des données de chaque feature avec des NaN
feature_cols = [c for c in df_dirty.columns if c != 'target']
for col in feature_cols:
    df_dirty.loc[df_dirty.sample(frac=0.05, random_state=42).index, col] = np.nan

print("Nombre total de valeurs manquantes générées :",
      df_dirty.isnull().sum().sum())

# --- PHASE 2 : DATA WRANGLING (NETTOYAGE) ---
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

print("Imputation terminée.")
print("Valeurs manquantes restantes :", X_clean.isnull().sum().sum())

# --- PHASE 3 : ANALYSE EXPLORATOIRE (EDA) ---
print("--- Statistiques Descriptives ---")
print(X_clean.describe())

# Exemple de visualisation 1 : Revenu médian vs Prix
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_clean['MedInc'], y=y, alpha=0.3)
plt.title("Relation entre Revenu Médian (MedInc) et Prix moyen des maisons")
plt.xlabel("MedInc (Revenu médian)")
plt.ylabel("Valeur moyenne des maisons (target)")
plt.show()

# Exemple de visualisation 2 : Matrice de corrélation
plt.figure(figsize=(10, 8))
corr = pd.concat([X_clean, y], axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de Corrélation (Features + cible)")
plt.show()

# --- PHASE 4 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

print("\nSéparation effectuée :")
print(f"Entraînement : {X_train.shape[0]} échantillons")
print(f"Test        : {X_test.shape[0]} échantillons")

# --- PHASE 5 : INTELLIGENCE ARTIFICIELLE (RANDOM FOREST REGRESSOR) ---
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- PHASE 6 : AUDIT DE PERFORMANCE ---
y_pred = model.predict(X_test)

from math import sqrt
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n>>> MSE  : {mse:.3f}")
print(f">>> RMSE : {rmse:.3f}")
print(f">>> MAE  : {mae:.3f}")
print(f">>> R²   : {r2:.3f}")

# Visualisation : Prédictions vs Réalité
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', label="Idéal"
)
plt.xlabel("Valeurs réelles (y_test)")
plt.ylabel("Prédictions (y_pred)")
plt.title("Random Forest - Prédictions vs Réalité (California Housing)")
plt.legend()
plt.show()

```

---

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### Le Problème Mathématique du "Vide"
Les algorithmes d’algèbre linéaire utilisés par les modèles de régression ne peuvent pas gérer la valeur NaN (Not a Number). Une seule valeur manquante dans une des features peut faire échouer l’entraînement ou fausser complètement les calculs de distances, de moyennes ou de splits dans les arbres.

### La Mécanique de l’Imputation
Nous utilisons SimpleImputer(strategy='mean').

1.  **L’Apprentissage (fit) :**  
    L’imputer scanne par exemple la colonne MedInc pour toutes les zones, calcule la moyenne du revenu médian, et stocke cette valeur. Il fait de même pour chaque feature (AveRooms, Population, etc.).

2.  **La Transformation (transform) :**  
    Lors de la transformation, dès qu’un trou (NaN) est rencontré dans une colonne, il est remplacé par la moyenne calculée à l’étape précédente pour cette colonne.

Au final, X_clean est une version “complète” du dataset, sans valeurs manquantes, compatible avec les algorithmes de Machine Learning.

### 💡 Le Coin de l’Expert (Data Leakage)
Attention : Dans un script pédagogique, on impute parfois avant le train_test_split pour simplifier. Dans un système industriel, c’est une *fuite de données* (Data Leakage).

*   Pourquoi ? Si la moyenne d’une feature est calculée sur tout le dataset (Train + Test), alors les valeurs du futur jeu de test ont indirectement influencé le nettoyage du Train.
*   La bonne pratique absolue :  
    *   D’abord séparer (Train/Test).  
    *   Fit l’imputer sur le Train uniquement.  
    *   Appliquer cette imputation au Test, sans recalculer les statistiques sur le Test.

---

## 4. Analyse Approfondie : Exploration (EDA)

C’est l’étape de "Profilage" du dataset immobilier.

### Décrypter .describe()
*   *Mean vs 50% (Médiane) :*  
    Pour des variables comme MedInc ou HouseAge, comparer la moyenne et la médiane permet de voir si la distribution est symétrique ou tirée par des quartiers très riches / très anciens.
*   *Std (Écart-type) :*  
    Mesure la dispersion des valeurs : un std élevé indique de fortes différences de revenu ou de densité entre quartiers, un std très faible signalerait une variable presque constante (peu utile pour le modèle).

### Corrélations et Structure Spatiale
En regardant la *heatmap de corrélation*, on peut observer :

*   Une forte corrélation positive entre MedInc (revenu médian) et la target (prix moyen des maisons), ce qui est intuitif : les zones plus riches ont des logements plus chers.
*   Des liens entre des variables comme AveRooms, AveBedrms et les prix, qui reflètent la taille moyenne des logements.
*   L’effet potentiel de la localisation (Latitude, Longitude) : en combinant ces variables avec la cible, on voit souvent que certaines zones géographiques (proche de la côte, par exemple) ont des prix systématiquement plus élevés.

---

## 5. Analyse Approfondie : Méthodologie (Split)

### Le Concept : La Garantie de Généralisation
Le but du modèle n’est pas de mémoriser les 20 640 échantillons historiques, mais d’être capable de prédire correctement les prix de logements dans de *nouveaux quartiers*.  
Pour cela, on sépare les données en Train/Test, et le Test est utilisé uniquement à la toute fin, comme un examen de généralisation.

### Les Paramètres sous le capot
train_test_split(test_size=0.2, random_state=42)

1.  *Le Ratio 80/20 :*  
    *   80 % des données servent à l’entraînement (le modèle apprend les patterns “prix = f(features)”).
    *   20 % sont gardés pour mesurer la performance sur des données jamais vues.

2.  **La Reproductibilité (random_state) :**  
    Fixer random_state=42 permet d’obtenir toujours la même séparation Train/Test, ce qui est essentiel pour comparer les résultats entre versions du modèle ou entre différents algorithmes.

---

## 6. FOCUS THÉORIQUE : L’Algorithme Random Forest 🌲

Pourquoi est-ce l’algorithme "couteau suisse" préféré des Data Scientists pour ce type de données tabulaires (revenu, densité, localisation, etc.) ?

### A. La Faiblesse de l’Individu (Arbre de Décision)
Un Arbre de Décision unique découpe l’espace des features en zones et affecte un prix moyen à chaque zone.

*   Problème : Il est *obsessif. Il peut se sur‑adapter au bruit d’un quartier très atypique (revenu extrêmement haut, prix extrême) et créer une règle très spécifique, ce qui conduit à une **haute variance* et des prédictions instables.

### B. La Force du Groupe (Bagging)

Random Forest signifie "Forêt Aléatoire". Il crée plusieurs dizaines (voire centaines) d’arbres.

1.  *Le Bootstrapping (Diversité des Échantillons) :*
    *   Chaque arbre s’entraîne sur un échantillon bootstrap différent des quartiers (avec tirage avec remise).
    *   Conséquence : Chaque arbre a une vision légèrement différente du marché immobilier californien.

2.  *Feature Randomness (Diversité des Questions) :*
    *   À chaque split, un arbre n’a accès qu’à un sous‑ensemble aléatoire des features (par exemple un sous‑ensemble des 8 variables).
    *   Conséquence : Certains arbres se spécialisent davantage sur les aspects géographiques (Latitude, Longitude), d’autres sur les variables socio‑démographiques (MedInc, Population), ce qui enrichit le panel d’“opinions”.

### C. Le Consensus (Vote / Moyenne)

Pour un nouveau quartier :

*   Chaque arbre propose un prix (prédiction de régression).
*   Le Random Forest prend la *moyenne* de ces prédictions.
*   Les erreurs individuelles des arbres (bruit) se compensent, ne laissant que la tendance lourde (le signal).

---

## 7. Analyse Approfondie : 


###  Les Métriques de Régression

On regarde plusieurs métriques complémentaires :

1.  *MSE (Mean Squared Error) :*  
    Moyenne des carrés des erreurs \((y_{réel} - y_{prédit})^2\). Très sensible aux grosses erreurs : un quartier fortement mal estimé pénalise beaucoup le MSE.

2.  *RMSE (Root Mean Squared Error) :*  
    Racine du MSE, exprimée dans la même unité que la target (centaines de milliers de dollars). Donne un ordre de grandeur de l’erreur moyenne en termes de prix.

3.  *MAE (Mean Absolute Error) :*  
    Moyenne des erreurs absolues \(|y_{réel} - y_{prédit}|\). Moins influencée par les outliers, elle donne une idée plus robuste de “combien” le modèle se trompe en moyenne par quartier.

4.  *R² (Coefficient de Détermination) :*  
    Mesure la proportion de la variance des prix expliquée par le modèle. Un R² proche de 1 signifie que le modèle explique bien les différences de prix entre quartiers ; un R² proche de 0 indique un modèle peu utile.


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
### Conclusion du Projet

Ce rapport montre que la Data Science ne s’arrête pas à model.fit(). C’est une chaîne de décisions cohérentes où :

*   La compréhension du métier (immobilier, prix, variabilité entre quartiers) guide le choix du dataset, des features et de la méthode de validation.
*   Le nettoyage, l’EDA, le split Train/Test et le choix d’un Random Forest robuste sont autant d’étapes critiques.
*   L’interprétation des métriques (MSE, RMSE, MAE, R²) et des visualisations permet de juger si le modèle est exploitable pour des applications réelles (agences, investisseurs, collectivités) ou s’il nécessite des itérations supplémentaires.

ch_california_housing(as_frame=True)

df = data.frame

df.rename(columns={'MedHouseVal': 'target'}, inplace=True)

print(f"📊 Dataset : {df.shape}")
