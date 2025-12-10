# IZLITNE MAROUANE

<img src="[https://image2url.com/images/1765362643039-fefb7bd2-5dd2-4bc3-9aa0-16fbe6d080c7.png](https://image2url.com/images/1765362786985-df3bb0b1-e113-40f7-a0cc-80d894c711cb.jpg)"
     alt="Logo MAROUANE IZLITNE"
     style="height:300px; margin-right:300px; float:left; border-radius:10px;">


**Numéro d'étudiant** : 22006529  
**Classe** : CAC2



---
# Détection de Fraude par Carte de Crédit 💳

## Description du Projet

Cette étude est une analyse et une modélisation d'un jeu de données de transactions par carte de crédit, avec l'objectif de développer un système automatisé capable d'identifier les opérations frauduleuses en temps réel.

### Problématique
Le problème est une tâche de **classification binaire** (Fraude vs. Non-Fraude). La difficulté principale réside dans l'**asymétrie extrême des classes** (Déséquilibre de classe), avec un ratio d'environ 99 transactions légitimes pour 1 transaction frauduleuse, ce qui rend la détection de la classe minoritaire (la fraude) particulièrement difficile pour les modèles classiques.

### Objectifs
L'objectif principal est de développer un modèle dont l'efficacité est mesurée par sa capacité à identifier la classe minoritaire. L'indicateur de performance clé est le **score ROC AUC**.


## Résumé des Résultats
Le projet a mis en œuvre une méthodologie rigoureuse incluant :

* **Ingénierie de caractéristiques temporelles** (Extraction de l'année, du mois, du jour de la semaine et de l'heure à partir de `TransactionDate`). 🕰️
* **Encodage One-Hot** pour les variables catégorielles (`TransactionType`, `Location`). 🏷️
* **Standardisation** pour les variables numériques (`Amount`, `MerchantID`). ⚖️
* **Modélisation et optimisation par grille** (`GridSearchCV`) en utilisant la pondération des classes (`class_weight='balanced'`) pour compenser le déséquilibre initial. 🧪
