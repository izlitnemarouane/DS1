# IZLITNE MAROUANE

<img src="https://image2url.com/images/1765362786985-df3bb0b1-e113-40f7-a0cc-80d894c711cb.jpg"
     alt="Logo marouane izlitne"
     style="height:300px; margin-right:300px; float:left; border-radius:10px;">

<br><br clear="left"/>
**Numéro d'étudiant** : 22006529  
**Classe** : CAC2

  
# Encadrant : Pr. Abderrahim larhlimi 

---
# 📊 ENCG SETTAT - Data Science & Modélisation Prédictive
## 🎯 **Mission : Prédire les prix immobiliers en Californie**
**Dataset : California Housing (scikit-learn)**
Lien du dataset : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing


<div style="text-align: center; font-size: 1.2em; color: #2c5aa0;">
🏠 <strong>Agences immobilières • Banques • Investisseurs</strong> 🏠
</div>


## Description du Projet
Cette étude déploie un pipeline complet de Machine Learning supervisé visant à prédire la valeur médiane des logements en Californie. S'appuyant sur le dataset California Housing (20 640 zones), l'analyse suit une méthodologie rigoureuse en quatre étapes :

Préparation des données : Pour simuler des conditions réelles, des valeurs manquantes ont été injectées puis traitées par imputation par la moyenne. Les variables ont ensuite été standardisées pour garantir la cohérence du modèle.

Exploration (EDA) : L'analyse a révélé que le revenu médian et la localisation géographique sont les principaux moteurs de la valeur immobilière.

Modélisation : Le choix s'est porté sur l'algorithme Random Forest Regressor (80% train / 20% test). Ce modèle a été privilégié pour sa robustesse face aux relations non linéaires et sa capacité à éviter le surapprentissage (overfitting).

Évaluation : Les performances ont été mesurées via le R² (précision globale) et le RMSE (erreur moyenne), permettant de valider la fiabilité des prédictions.
## Problématique
Le problème est une tâche de **régression supervisée**. La difficulté principale réside dans la **variabilité spatiale extrême** des prix (zones côtières vs intérieures) et la **présence de valeurs aberrantes** (quartiers très chers), combinée à des données potentiellement incomplètes en contexte réel.

## Objectifs
L'objectif principal est de développer un modèle dont l'efficacité est mesurée par sa capacité à expliquer la variance des prix (**R²**) et à minimiser l'erreur prédictive en unités métier (**RMSE** en centaines de milliers de dollars).

## Résumé des Résultats
Le projet a mis en œuvre une méthodologie rigoureuse incluant :

**Simulation de données réalistes** : Injection de 5% de valeurs manquantes (NaN) 🕳️

**Imputation par moyenne** (`SimpleImputer`) pour conserver toutes les observations 📊 

**Standardisation** des features (`StandardScaler`) pour stabiliser l'apprentissage ⚖️

**Analyse exploratoire** : Distribution des prix + corrélations (MedInc dominant) 📈  

**Modélisation Random Forest** (`RandomForestRegressor`, 100 arbres) avec split 80/20 🔀 

**Métriques clés** : R² ≈ 0.80-0.85, RMSE ≈ 0.5 ($50k d'erreur moyenne) 🎯  

**Meilleures features** : `MedInc` (revenu médian), coordonnées géographiques 🗺️
