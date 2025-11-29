# Compte rendu
## Analyse Prédictive des Ventes — Business_Sales(Dataset)2025

**Date :** 29 Novembre 2025

***

## À propos du jeu de données

Le dataset **Business_Sales(Dataset)2025** (Kaggle) contient des données commerciales pour 2025 — variables marketing, prix, remises, indicateurs opérationnels, indicateurs macroéconomiques et le volume de ventes (`Sales`) en tant que variable cible.  
Chaque ligne représente un enregistrement magasin/produit/période et permet d'analyser et de prédire les volumes de ventes selon des facteurs internes et externes.

***

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)  
2. [Analyse Exploratoire des Données (Data Analysis)](#2-analyse-exploratoire-des-données-data-analysis)  
   - [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)  
   - [Prétraitement et Ingénierie de Caractéristiques](#22-prétraitement-et-ingénierie-de-caractéristiques)  
   - [Gestion des Valeurs Manquantes](#23-gestion-des-valeurs-manquantes)  
   - [Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)  
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)  
   - [Séparation des Données (Data Split)](#31-séparation-des-données-data-split)  
   - [Modèles de Régression Testés](#32-modèles-de-régression-testés)  
4. [Résultats et Comparaison des Modèles](#4-résultats-et-comparaison-des-modèles)  
   - [Régression Linéaire](#41-régression-linéaire)  
   - [Régression Polynomiale](#42-régression-polynomiale)  
   - [Régression par Arbre de Décision](#43-régression-par-arbre-de-décision)  
   - [Régression par Forêt Aléatoire](#44-régression-par-forêt-aléatoire)  
   - [Régression SVR (Support Vector Regression)](#45-régression-svr-support-vector-regression)  
   - [Tableau Comparatif des Performances](#46-tableau-comparatif-des-performances)  
5. [Analyse des Résultats et Recommandations](#5-analyse-des-résultats-et-recommandations)  
6. [Conclusion](#6-conclusion)  
7. [Code Python utilisé (extraits clés) et lien de téléchargement du code complet](#7-code-python-utilisé-extraits-clés-et-lien-de-téléchargement-du-code-complet)

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
