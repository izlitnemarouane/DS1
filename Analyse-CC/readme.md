# IZLITNE MAROUANE

<img src="IZLITNE MAROUANE.jpg" width="300" align="left" style="margin-right: 30px; border-radius: 10px;"/>

<br><br clear="left"/>



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

## Installation

Pour exécuter ce projet localement, clonez le dépôt et installez les dépendances listées dans `requirements.txt` :

```bash
git clone <(https://github.com/izlitnemarouane/DS1/edit/main/Analyse-CC)>
DS1 <Analyse-CC>
pip install -r requirements.txt

```
## Résumé des Résultats
Le projet a mis en œuvre une méthodologie rigoureuse incluant :

* **Ingénierie de caractéristiques temporelles** (Extraction de l'année, du mois, du jour de la semaine et de l'heure à partir de `TransactionDate`). 🕰️
* **Encodage One-Hot** pour les variables catégorielles (`TransactionType`, `Location`). 🏷️
* **Standardisation** pour les variables numériques (`Amount`, `MerchantID`). ⚖️
* **Modélisation et optimisation par grille** (`GridSearchCV`) en utilisant la pondération des classes (`class_weight='balanced'`) pour compenser le déséquilibre initial. 🧪
