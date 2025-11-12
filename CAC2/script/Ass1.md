<img src="me.jpg" style="height:464px;margin-right:432px"/>

## IZLITNE MAROUANE 

## N appogee 22006529

# description 
La base de données Heart Disease regroupe des informations médicales destinées à l’analyse et à la prédiction des maladies cardiaques. Elle provient du UCI Machine Learning Repository et contient des observations de patients issus de plusieurs centres hospitaliers (Cleveland, Hongrie, Suisse et Californie). L’objectif principal de cette base est de déterminer la présence ou l’absence d’une maladie cardiaque à partir de caractéristiques cliniques et physiologiques du patient.

Les variables incluent des données telles que l’âge, le sexe, le type de douleur thoracique, la pression artérielle au repos, le taux de cholestérol, la glycémie à jeun, les résultats de l’électrocardiogramme, la fréquence cardiaque maximale atteinte, ainsi que l’existence d’angine induite par l’effort. La variable cible indique si le patient est atteint d’une maladie cardiaque (1) ou non (0).

Cette base contient environ 303 enregistrements (dans la version la plus utilisée, celle de Cleveland) et 14 attributs principaux. Elle est fréquemment utilisée pour l’entraînement et l’évaluation de modèles de classification en apprentissage automatique, notamment pour développer des systèmes d’aide à la décision médicale capables de détecter précocement les risques cardiaques.

# Analyse descriptive et statistique de la base "Heart Disease" (fusion UCI)

## 1. Origine de la base
Cette base provient du *Heart Disease Data Set* du *UCI Machine Learning Repository*.  
Elle combine les données de quatre centres médicaux : *Cleveland, **Hungarian, **Switzerland, et **VA Long Beach*.  
Les auteurs originaux incluent Janosi, Steinbrunn, Pfisterer, Detrano, et collaborateurs.

## 2. Fichiers fusionnés
Les fichiers suivants ont été fusionnés :
- processed.cleveland.data
- processed.hungarian.data
- processed.switzerland.data
- processed.va.data

Taille finale : *920 lignes × 15 colonnes*.

## 3. Variables principales
| Variable | Signification | Type |
|-----------|---------------|------|
| age | Âge du patient | Numérique |
| sex | Sexe (1 = homme, 0 = femme) | Binaire |
| cp | Type de douleur thoracique | Catégorique |
| trestbps | Pression artérielle au repos | Numérique |
| chol | Cholestérol sérique (mg/dl) | Numérique |
| fbs | Taux de sucre à jeun >120 mg/dl (1 = vrai, 0 = faux) | Binaire |
| restecg | Résultat de l’électrocardiogramme | Catégorique |
| thalach | Fréquence cardiaque max atteinte | Numérique |
| exang | Angine provoquée par exercice (1 = oui, 0 = non) | Binaire |
| oldpeak | Dépression du segment ST | Numérique |
| slope | Pente du segment ST | Catégorique |
| ca | Nombre de vaisseaux majeurs colorés | Numérique |
| thal | Type de thalassémie | Catégorique |
| target | Présence de maladie (0 = non, 1-4 = oui) | Cible |

## 4. Statistiques descriptives
- Total : 920 observations.  
- Variables avec *beaucoup de valeurs manquantes* :
  - ca : 611 valeurs manquantes
  - thal : 486 valeurs manquantes
  - slope : 309 valeurs manquantes
- Moyenne d’âge : autour de 54 ans.  
- target_bin : 509 malades (1) / 411 non-malades (0).

## 5. Corrélations clés
| Variables | Corrélation absolue |
|------------|---------------------|
| ca – target | 0.52 |
| oldpeak – target | 0.44 |
| thal – target | 0.44 |
| cp – target | 0.40 |

Ces corrélations montrent des associations significatives entre certains indicateurs cliniques et la présence de maladie cardiaque.

## 6. Graphiques produits
1. *Histogramme de l’âge* — distribution globale des patients.  
2. *Répartition du sexe* — proportion hommes/femmes.  
3. *Boxplot du cholestérol* selon target_bin.  
4. *Nuage de points âge vs fréquence cardiaque max (thalach)*.  
5. *Matrice de corrélation* entre variables quantitatives.

## 7. Interprétation rapide
- L’ensemble fusionné est équilibré (55% malades environ).  
- ca et thal nécessitent une imputation.  
- chol, oldpeak, et cp semblent informatifs pour la prédiction.  
- L’analyse de corrélation et les boxplots confirment certaines tendances connues médicalement.

## 8. Recommandations
1. Traiter les valeurs manquantes (imputation KNN ou médiane).  
2. Encoder correctement les variables catégoriques (cp, thal, slope, restecg).  
3. Évaluer la robustesse par centre médical (source_file).  
4. Construire un modèle de classification (régression logistique, arbre, etc.).

## 9. Références
- *UCI Machine Learning Repository – Heart Disease Data Set*  
  <https://archive.ics.uci.edu/ml/datasets/heart+disease>

---

*Fichier fusionné disponible :* heart_disease_merged.csv  
*Graphiques générés :*  
- plot_age_hist.png  
- plot_sex_bar.png  
- plot_chol_box.png  
- plot_age_thalach_scatter.png  
- plot_corr_matrix.png

<img src="data.png" style="height:464px;margin-right:432px"/>

