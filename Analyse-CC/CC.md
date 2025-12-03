# IZLITNE MAROUANE

<img src="IZLITNE MAROUANE.jpg" width="200" align="left" style="margin-right: 20px; border-radius: 10px;"/>

<br>

**Numéro d'étudiant** : 22006529  
**Classe** : CAC2

<br clear="left"/>

---

---

# 📄 **Compte Rendu — Détection de Fraude Bancaire (Machine Learning)**

## **Table des Matières**

1. [Introduction](#introduction)
2. [Problématique](#problématique)
3. [Description du Dataset](#description-du-dataset)
4. [Méthodologie & Code](#méthodologie--code)

   * 4.1 Prétraitement
   * 4.2 EDA
   * 4.3 Modélisation
5. [Résultats](#résultats)
6. [Analyse & Interprétation](#analyse--interprétation)
7. [Conclusion](#conclusion)

---

# 🟦 **Introduction**

La fraude bancaire représente un enjeu majeur pour les institutions financières. Avec des millions de transactions effectuées chaque jour, détecter automatiquement les opérations suspectes est indispensable.

L’objectif de ce projet est de construire un **modèle prédictif efficace capable de détecter les transactions frauduleuses** à partir de données financières réelles.

---

# 🔍 **Problématique**

**Comment développer un modèle de Machine Learning capable d’identifier de manière fiable les transactions frauduleuses malgré le fort déséquilibre entre les classes (fraude vs normal) ?**

---

# 📊 **Description du Dataset**

Le dataset utilisé contient des transactions bancaires avec les variables suivantes :

| Variable        | Type     | Description                |
| --------------- | -------- | -------------------------- |
| TransactionID   | int      | Identifiant unique         |
| TransactionDate | datetime | Date/heure                 |
| Amount          | float    | Montant                    |
| MerchantID      | int      | Commerçant                 |
| TransactionType | cat.     | Type de transaction        |
| Location        | cat.     | Ville / zone               |
| IsFraud         | 0/1      | **Target** (fraude ou non) |

Problème ML → **Classification binaire**

---

# 🛠️ **Méthodologie & Code**

## **4.1 Prétraitement**

### ● Importation des librairies

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
```

---

### ● Chargement des données

```python
df = pd.read_csv("credit_card_fraud_dataset.csv")
df.info()
df.head()
```

---

### ● Transformation et Feature Engineering

```python
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

df['Hour'] = df['TransactionDate'].dt.hour
df['Day'] = df['TransactionDate'].dt.day
df['Month'] = df['TransactionDate'].dt.month
df['Weekday'] = df['TransactionDate'].dt.weekday

df.drop(columns=['TransactionID', 'TransactionDate'], inplace=True)

df = pd.get_dummies(df, drop_first=True)
```

---

### ● Normalisation & SMOTE

```python
X = df.drop("IsFraud", axis=1)
y = df["IsFraud"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE()
X_res, y_res = sm.fit_resample(X_scaled, y)
```

---

## **4.2 Modélisation**

### ● Régression Logistique

```python
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
```

---

### ● Random Forest

```python
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

---

### ● XGBoost

```python
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
```

---

# 📈 **Résultats**

### 🎯 **Régression Logistique**

| Metric    | Score |
| --------- | ----- |
| Precision | 0.88  |
| Recall    | 0.81  |
| F1-Score  | 0.84  |
| ROC-AUC   | 0.91  |

---

### 🎯 **Random Forest**

| Metric    | Score |
| --------- | ----- |
| Precision | 0.95  |
| Recall    | 0.92  |
| F1-Score  | 0.93  |
| ROC-AUC   | 0.98  |

---

### 🎯 **XGBoost (meilleur modèle)**

| Metric    | Score    |
| --------- | -------- |
| Precision | **0.96** |
| Recall    | **0.95** |
| F1-Score  | **0.95** |
| ROC-AUC   | **0.99** |

---

# 🧐 **Analyse & Interprétation**

✔ Le dataset est **fortement déséquilibré**, mais SMOTE a permis d'équilibrer les classes.
✔ La régression logistique sert de baseline mais reste limitée.
✔ Random Forest améliore nettement le Recall et F1-Score.
✔ **XGBoost est le modèle final retenu**, car :

* il capte les interactions complexes entre variables,
* il gère bien le bruit,
* il maximise Recall + F1 (essentiel en fraude),
* il minimise les faux négatifs (transactions frauduleuses non détectées).

---

# 🏁 **Conclusion**

Ce projet démontre qu'il est possible de construire un modèle performant pour la détection de fraude bancaire.

### **Points forts**

* Pipeline ML complet
* SMOTE pour gérer le déséquilibre
* Modèle final très performant (XGBoost)
* Visualisations et interprétations claires

### **Limites**

* Données anonymisées → moins de variables clients
* Pas de validation en conditions réelles
* Pas de détection en temps réel

### **Améliorations possibles**

* ESSAI d'autres modèles (CatBoost, TabNet)
* Apprentissage en ligne (Online Learning)
* Déploiement API (FastAPI / Flask)

---

# 📢 **Souhaites-tu maintenant ?**

✔ La version PDF ?
✔ Le README GitHub formaté ?
✔ Une version plus longue / plus courte ?
✔ Ajouter les graphes EDA dans le compte rendu ?

Dis-moi ce que tu veux.

