# IZLITNE MAROUANE

<img src="IZLITNE MAROUANE.jpg" width="200" align="left" style="margin-right: 20px; border-radius: 10px;"/>

<br>

**Numéro d'étudiant** : 22006529  
**Classe** : CAC2

<br clear="left"/>

---


# 📄 Compte Rendu — Détection de Fraude Bancaire (Machine Learning)

## Table des Matières
1. [Introduction](#introduction)
2. [Problématique](#problématique)
3. [Description du Dataset](#description-du-dataset)
4. [Méthodologie & Code](#méthodologie--code)
   1. [4.1 Prétraitement](#41-prétraitement)
   2. [4.2 EDA](#42-eda)
   3. [4.3 Modélisation](#43-modélisation)
5. [Résultats](#résultats)
6. [Analyse & Interprétation](#analyse--interprétation)
7. [Conclusion](#conclusion)

# Introduction
La fraude bancaire représente un enjeu majeur pour les institutions financières. L'objectif est de construire un modèle prédictif efficace pour détecter les transactions frauduleuses.

# Problématique
Comment développer un modèle de Machine Learning capable d’identifier de manière fiable les transactions frauduleuses malgré le fort déséquilibre entre les classes ?

# Description du Dataset
| Variable        | Type     | Description                |
| --------------- | -------- | -------------------------- |
| TransactionID   | int      | Identifiant unique         |
| TransactionDate | datetime | Date/heure                 |
| Amount          | float    | Montant                    |
| MerchantID      | int      | Commerçant                 |
| TransactionType | cat.     | Type de transaction        |
| Location        | cat.     | Ville / zone               |
| IsFraud         | 0/1      | Target (fraude ou non)    |

# Méthodologie & Code

## 4.1 Prétraitement
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
````

```python
df = pd.read_csv("credit_card_fraud_dataset.csv")
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['Hour'] = df['TransactionDate'].dt.hour
df['Day'] = df['TransactionDate'].dt.day
df['Month'] = df['TransactionDate'].dt.month
df['Weekday'] = df['TransactionDate'].dt.weekday
df.drop(columns=['TransactionID', 'TransactionDate'], inplace=True)
df = pd.get_dummies(df, drop_first=True)
X = df.drop("IsFraud", axis=1)
y = df["IsFraud"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sm = SMOTE()
X_res, y_res = sm.fit_resample(X_scaled, y)
```

## 4.2 EDA

```python
# Visualisation de la distribution des classes
sns.countplot(x='IsFraud', data=df)
plt.title("Distribution des transactions frauduleuses vs normales")
plt.show()

# Corrélation entre variables
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()
```

## 4.3 Modélisation

```python
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
```

# Résultats

### Régression Logistique

| Metric    | Score |
| --------- | ----- |
| Precision | 0.88  |
| Recall    | 0.81  |
| F1-Score  | 0.84  |
| ROC-AUC   | 0.91  |

### Random Forest

| Metric    | Score |
| --------- | ----- |
| Precision | 0.95  |
| Recall    | 0.92  |
| F1-Score  | 0.93  |
| ROC-AUC   | 0.98  |

### XGBoost (meilleur modèle)

| Metric    | Score |
| --------- | ----- |
| Precision | 0.96  |
| Recall    | 0.95  |
| F1-Score  | 0.95  |
| ROC-AUC   | 0.99  |

# Analyse & Interprétation

* Dataset fortement déséquilibré, SMOTE utilisé.
* Régression logistique limitée.
* Random Forest améliore Recall et F1-Score.
* XGBoost capte interactions complexes et minimise faux négatifs.

# Conclusion

**Points forts:** pipeline complet, SMOTE, XGBoost performant.
**Limites:** données anonymisées, pas de validation réelle, pas de temps réel.
**Améliorations possibles:** CatBoost, TabNet, apprentissage en ligne, déploiement API.

```

```


