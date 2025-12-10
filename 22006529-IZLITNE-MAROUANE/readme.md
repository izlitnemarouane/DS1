# IZLITNE MAROUANE

<img src="IZLITNE MAROUANE.jpg" width="300" align="left" style="margin-right: 30px; border-radius: 10px;"/>

<br><br clear="left"/>



**Numéro d'étudiant** : 22006529  
**Classe** : CAC2


# Prédiction des Prix Immobiliers en Californie 🏡

## Description du Projet
Cette étude est une analyse et une modélisation du dataset **California Housing** (scikit-learn), avec l'objectif de développer un système automatisé capable de prédire la **valeur médiane des maisons** par bloc géographique en Californie.

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
