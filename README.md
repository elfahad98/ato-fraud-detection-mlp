# Détection d'Account Takeover (ATO) — Machine Learning avec scikit-learn (Python)

##  Objectif du projet
Ce projet vise à **détecter les tentatives de prise de contrôle de compte (Account TakeOver)** à partir de logs d’authentification utilisateur.  
Les données proviennent du dataset **RBA Dataset** de Kaggle, comprenant plus de **33 millions de lignes**, simulant des connexions légitimes et malveillantes dans un contexte de cybersécurité.
L’objectif est de construire un **modèle robuste et interprétable** capable d’identifier les comportements anormaux tout en limitant les faux positifs (FPR ≈ 1%).

---

##  Stack technique
| Domaine                 | Outils |
|-------------------------|--------|
| Préparation & analyse   | **Pandas**, **NumPy** |
| Modélisation            | **scikit-learn** (MLPClassifier, Logistic Regression, XGBoost) |
| Évaluation              | **PR-AUC**, **ROC-AUC**, **Recall@1%FPR**, matrice de confusion |
| Interprétabilité        | **SHAP** |
| Visualisation           | **Matplotlib**,**Seaborn** |
| Environnement           | **Jupyter Notebook**, **Python 3.11** |

---

##  Jeu de données & échantillonnage
- **Source :** RBA Dataset (Kaggle) — plus de **33 millions de connexions simulées** entre utilisateurs légitimes et attaques ATO.
- **Objectif :** identifier les connexions frauduleuses à partir des logs d’authentification.
- **Sous-échantillon utilisé (≈ 300 000 lignes)** :
  - Conservation **de toutes les fraudes** (141 échantillons).
  - **Échantillonnage aléatoire de 1%** des connexions normales (~312 000).
  - Chargement progressif par **chunks de 200 000 lignes** pour éviter la surcharge mémoire.
- Cet échantillon est utilisé **pour l’ensemble des étapes** :  
  **EDA**, **prétraitement**, **feature engineering**, **modélisation** et **interprétabilité**.
- Le modèle final est calibré pour **1% de faux positifs (FPR)** et évalué sur un **jeu de test distinct**.

---

##  Méthodologie (6 étapes)

1) **Chargement & échantillonnage (~300k)**
   - **Conserver toutes les fraudes** et **échantillonner ~1%** des connexions normales.
   - Constituer un jeu de travail équilibré pour l’EDA et l’entraînement.

2) **Nettoyage & préparation**
   - Contrôle du typage, détection d’incohérences et doublons.
   - Gestion des valeurs manquantes (imputation sobre + indicateurs de présence).
   - Regroupement des **catégories rares** (ex. pays/devices) pour limiter la variabilité.
   - Mise à l’échelle cohérente des variables numériques.

3) **EDA orientée risque**
   - Mesure de la **prévalence ATO** (globale & par segments : device, pays, créneaux horaires).
   - Analyse des **comportements inhabituels** (ex. changement soudain de pays/device, délais entre connexions).
   - Identification des **facteurs et patterns** corrélés au risque (réseau/ASN, RTT, succès/échec login).
   - Formulation d’hypothèses qui guident le feature engineering.

4) **Feature Engineering**
   - **Temporel** : rythme d’activité, jour/heure, week-end.
   - **Comportement utilisateur** : transitions (nouveau pays/device/ASN), **délai depuis la dernière connexion**.
   - **Réseau** : signaux de latence/RTT et fournisseur (ASN).
   - **Géographie & device** : regroupements lisibles, réduction des modalités rares.
   - Encodage des catégorielles et harmonisation des échelles .

5) **Modélisation & déséquilibre**
   - **Split stratifié 80/20** (train/test) et **pondération des classes** pour compenser le fort déséquilibre.
   - **Benchmark de trois modèles :**
     - **Régression logistique** → modèle **baseline** (linéaire, simple et interprétable).  
     - **XGBoost** → modèle **arborescent**, robuste aux interactions complexes.  
     - **MLPClassifier** → réseau de neurones **peu profond** (128–64 neurones) choisi comme **modèle final**.
   - Sélection du meilleur compromis sur **PR-AUC**, **ROC-AUC** et **Recall@1%FPR**, adaptés au contexte de détection rare.


6) **Optimisation, seuil & sélection finale**
   - Ajustements ciblés (régularisation, taille du réseau, gestion des catégories rares) pour **réduire l’overfit**.
   - **Calibration du seuil métier** sur **TRAIN** pour viser **~1% de faux positifs**, puis validation sur **TEST**.
   - **Interprétabilité** : importance globale et impacts locaux (SHAP) pour expliciter les facteurs de risque.
   - **Export** des artefacts (pipeline + seuil + méta) pour un usage reproductible.

---

##  Résultats principaux

| Indicateur | Train | Test |
|-------------|-------|------|
| **AUC-ROC** | 0.9949 | 0.9696 |
| **PR-AUC** | 0.7585 | 0.7092 |
| **Recall@1%FPR** | 0.991 | 0.893 |
| **Precision@1%FPR** | 0.037 | 0.042 |

> **Seuil choisi : 0.0191 (≈ 1% de faux positifs)**  
> → Bon équilibre entre rappel élevé et faible taux de fausses alertes.

---

## Visualisations — Analyse

| Graphique | Analyse |
|---|---|
| ![Courbe Precision–Recall](screenshots/PR.png) | **AP ≈ 0.709**, très au-dessus de la baseline ≈ **0.00045** (prévalence).<br>Le modèle classe **très bien** les attaques malgré l’ultra-déséquilibre.<br> À **recall ≈ 0.89**, la **precision ≈ 3.7%** (coût d’alerte acceptable pour du **screening large**). |
| ![Courbe ROC](screenshots/ROC.png) | **AUC ≈ 0.97** (excellente séparation globale). Au point métier **@ ~1% FPR**, on lit **TPR ≈ 0.893**. |
| ![Matrice de confusion](screenshots/mat_confus.png) | Seuil calibré pour **~1% FPR** (θ ≈ 0.0191).<br>**TP=25**, **FP=645**, **FN=3**, **TN=61 894** ⇒<br> **Recall = 0.893**, **Precision = 0.037**, **FPR ≈ 0.0103**.<br>Lecture métier : on **capte presque toutes les fraudes** (3 manquées) au prix d’**~645 alertes** à vérifier. |
| ![SHAP summary](screenshots/shap.png) | Facteurs qui **poussent le score** : **ASN** (29695, 29492, 393398), **Country** (US/NO/PL/DE),<br> **Device Type** (mobile/desktop), **Login Successful**.<br>→ Risque lié aux **changements d’environnement** (réseau/appareil) et à certains **fournisseurs**. |

---

##  Interprétation SHAP

Les attributs les plus influents sur la détection de fraude :
- **ASN (réseau d’origine)** : certains ASN associés à des anomalies.
- **Login Successful** : indicateur clé d’accès suspect.
- **Country (US, NO, PL, DE)** : importance géographique dans la détection.
- **Device Type** : différences notables entre mobile, desktop, tablet.

---

## 👤 Auteur

Projet réalisé par **COMBO El-Fahad** – Université de Caen (2025).  
Contact : `el-fahad.combo@etu.unicaen.fr`

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier `LICENSE`.
