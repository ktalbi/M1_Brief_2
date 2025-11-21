
# M1_Brief2 : API, Réentraînement, Streamlit & MLflow

## Objectif du Projet

Ce projet met en place une solution combinant :

- **Une API FastAPI** permettant :
  - l’entraînement d’un modèle (`/train`)
  - la prédiction à partir de CSV (`/predict`)
  - un endpoint de santé (`/health`)
- **Une application Streamlit** pour interagir facilement avec l’API
- **Le suivi des modèles et métriques via MLflow**
- **La conteneurisation complète via Docker Compose**
- **Un pipeline de tests unitaires** pour assurer la robustesse

L’approche couvre tout le cycle de vie du modèle :  
**ingestion → preprocessing → entraînement → suivi → prédiction → réentraînement → déploiement**.

## 1. Architecture Globale

```
.
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── dockerfile
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── dockerfile
├── modules/
│   ├── preprocessing.py
│   └── evaluate.py
├── models/
│   ├── preprocessor.pkl
│   └── mlflow_model_uri.txt
├── logs/
├── mlruns/               
├── tests/
│   ├── test_api_endpoints.py
│   ├── test_streamlit_app.py
├── docker-compose.yml
└── README.md
```

## 2. Fonctionnalités Principales

### 2.1 API FastAPI

L’API expose trois endpoints principaux :

#### **POST `/train`**
- reçoit un CSV contenant **INPUT + TARGET**
- prétraite via `preprocessor.pkl`
- entraîne un modèle TensorFlow/Keras
- loggue métriques + artefacts dans **MLflow**
- met à jour le `mlflow_model_uri.txt`
- recharge le nouveau modèle en mémoire

#### **POST `/predict`**
- reçoit un CSV contenant **seulement les features**
- applique le préprocesseur
- renvoie les prédictions au format JSON

#### **GET `/health`**
- vérifie disponibilité du modèle
- renvoie version du modèle + timestamp

### 2.2 Suivi MLflow

Chaque entraînement loggue :

| Métrique | Description |
|---------|-------------|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **R²** | Qualité de variance expliquée |

Les modèles sont sauvegardés en tant qu’artefacts MLflow (format sklearn).

L’URI est stocké dans :

```
models/mlflow_model_uri.txt
```

### 2.3 Application Streamlit

L’app fournit :

- un sélecteur de fichier CSV
- un bouton **Entraîner le modèle**
- un bouton **Prédire**
- affichage des prédictions
- configuration de l’URL de l’API (local ou Docker)

L’URL par défaut est :

- en développement local : `http://localhost:8000`
- en Docker Compose : `http://api:8000`

## 3. Déploiement via Docker Compose

Lancement :

```bash
docker compose up --build
```

Accès :

- API → http://localhost:8000/docs  
- Streamlit → http://localhost:8501

## 4. Tests Unitaires

Les tests se trouvent dans `tests/` et couvrent :

- `/health`
- `/train`
- `/predict`
- import Streamlit
- génération de CSV synthétiques

Lancement :

```bash
pytest -q
```

## 5. Lancement Local (hors Docker)

API :

```bash
uvicorn api.main:app --reload --port 8000
```

Streamlit :

```bash
cd app
streamlit run app.py
```

## 6. Exemple de CSV attendu

### Pour /train :

```
prenom,age,taille,poids,sexe,sport_licence,niveau_etude,region,smoker,nationalité_francaise,revenu_estime_mois,montant_pret
Zoé,57,161.7,56.3,F,oui,bac+2,Normandie,non,non,1369,500.0
```

### Pour /predict :

```
prenom,age,taille,poids,sexe,sport_licence,niveau_etude,region,smoker,nationalité_francaise,revenu_estime_mois
Zoé,57,161.7,56.3,F,oui,bac+2,Normandie,non,non,1369
```

## 7. Conclusion

Ce projet propose une architecture robuste:
- API + Streamlit
- MLflow pour le suivi
- Pipeline d’entraînement / réentraînement
- Docker pour le déploiement
- Tests unitaires pour la stabilité
