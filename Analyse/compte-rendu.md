# IZLITNE MAROUANE

<img src="IZLITNE MAROUANE.jpg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>


**Numéro d'étudiant** : 22006529
**Classe** : CAC2

<br clear="left"/>

---

# Compte rendu
## Analyse Avancée et Prédiction du Volume de Ventes (Sales Volume) par Modèles de Régression

**Date :** 26 Novembre 2025

---

# À propos du jeu de données :

Ce fichier contient des données de ventes commerciales sur **plusieurs périodes**, incluant les caractéristiques des produits, les stratégies de promotion et les performances saisonnières. Chaque ligne représente un enregistrement de produit et inclut des informations sur sa catégorie, son prix, sa position dans le magasin/site web, et le volume de ventes réalisé.

Ce jeu de données est essentiel pour la **planification des stocks (Inventory Management)** et l'**optimisation des campagnes marketing**, car il permet de modéliser l'impact de divers facteurs logistiques et promotionnels sur la demande des produits.

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Analyse Exploratoire des Données (Data Analysis)](#2-analyse-exploratoire-des-données-data-analysis)
    * [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)
    * [Prétraitement et Ingénierie de Caractéristiques](#22-prétraitement-et-ingénierie-de-caractéristiques)
    * [Gestion des Valeurs Manquantes](#23-gestion-des-valeurs-manquantes)
    * [Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)
    * [Séparation des Données (Data Split)](#31-séparation-des-données-data-split)
    * [Modèles de Régression Testés et Ensembles](#32-modèles-de-régression-testés-et-ensembles)
4. [Résultats et Comparaison des Modèles](#4-résultats-et-comparaison-des-modèles)
    * [Performance des Modèles Linéaires](#41-performance-des-modèles-linéaires)
    * [Performance des Modèles Arborescents](#42-performance-des-modèles-arborescents)
    * [Modèle Optimal : Gradient Boosting](#43-modèle-optimal--gradient-boosting)
    * [Tableau Comparatif des Performances](#44-tableau-comparatif-des-performances)
5. [Analyse des Résultats et Recommandations](#5-analyse-des-résultats-et-recommandations)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction et Contexte

Ce rapport documente la démarche Data Science appliquée à l'analyse et à la prédiction du **Volume de Ventes** d'un portefeuille de produits. L'objectif est de trouver le modèle de régression le plus performant pour fournir une anticipation fiable de la demande.

Nous avons parcouru toutes les étapes du cycle de vie des données, de l'exploration approfondie (EDA) à la modélisation prédictive, en nous concentrant sur les **modèles d'ensemble** qui se sont révélés les plus efficaces.

---

## 2. Analyse Exploratoire des Données (Data Analysis)

### 2.1 Chargement et Structure du Dataset

Le jeu de données `Business_sales_EDA.csv` est la base de notre analyse.

* **Nombre d'observations ($N$) :** (À compléter) L'échelle des données est de plusieurs milliers d'enregistrements.
* **Nombre de variables ($d$) :** (À compléter) Colonnes incluant la catégorie, le prix, la promotion et le volume de ventes.

**Variable cible ($Y$) :** `Sales Volume` (Volume de Ventes).

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Chargement des données (Le séparateur ';' est souvent utilisé dans les fichiers européens)
df = pd.read_csv('Business_sales_EDA.csv', sep=';')
print("========= Résumé du Dataset =========")
print(f"Dimensions : {df.shape}")
df.info()
print("\n========= Premiers échantillons =========")
print(df.head())
