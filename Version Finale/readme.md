# IZLITNE MAROUANE

<img src="https://image2url.com/images/1765362786985-df3bb0b1-e113-40f7-a0cc-80d894c711cb.jpg"
     alt="Logo marouane izlitne"
     style="height:300px; margin-right:300px; float:left; border-radius:10px;">

<br><br clear="left"/>



**Numéro d'étudiant** : 22006529  
**Classe** : CAC2


# Prédiction des Prix Immobiliers en Californie 🏡

## Description du Projet
L’analyse « Modélisation et Prédiction des Prix Immobiliers en Californie par Forêts Aléatoires » présente un pipeline complet de régression supervisée pour estimer la valeur médiane des maisons (en centaines de milliers de dollars) à partir du dataset California Housing de scikit‑learn, qui décrit 20 640 zones géographiques au moyen de 8 variables socio‑démographiques et géographiques. Après chargement des données sous forme de DataFrame, des données « sales » sont simulées par injection de valeurs manquantes, puis un nettoyage est appliqué via une imputation par la moyenne et une standardisation des variables explicatives afin d’obtenir un jeu de données homogène et exploitable. Une analyse exploratoire est ensuite menée (statistiques descriptives, étude de la distribution de la cible, corrélations entre features et prix) pour identifier les facteurs qui influencent le plus la valeur des logements, en particulier le revenu médian et la localisation. La partie modélisation s’appuie sur un Random Forest Regressor entraîné sur un découpage train/test (80/20), choisi pour sa robustesse, sa capacité à capturer des relations non linéaires et à limiter le surapprentissage grâce au bagging et à la sélection aléatoire de variables. Les performances sont évaluées à l’aide du R² et du RMSE, complétés par un graphique « valeurs réelles vs prédictions », ce qui permet de quantifier l’erreur moyenne en unités métier et de vérifier visuellement la qualité de calibration du modèle dans une optique d’aide à la décision pour les acteurs immobiliers et financiers. Le dataset utilisé est accessible via la documentation officielle :
Lien du dataset : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing
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
