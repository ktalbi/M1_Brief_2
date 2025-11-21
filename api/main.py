from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import joblib
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from os.path import join as join
import mlflow
import mlflow.sklearn

from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from models.models import create_nn_model, train_model, model_predict
import os
# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Colonnes d'entrée (features)
INPUT_COLUMNS = [
    "prenom",
    "age",
    "taille",
    "poids",
    "sexe",
    "sport_licence",
    "niveau_etude",
    "region",
    "smoker",
    "nationalité_francaise",
    "revenu_estime_mois",
]

# Colonne cible
TARGET_COLUMN = "montant_pret"
_model = None
_model_version = None
MLFLOW_MODEL_URI_FILE = join("models", "mlflow_model_uri.txt")


# Charger le préprocesseur
preprocessor = joblib.load(join('models','preprocessor.pkl'))


MLFLOW_URI_PATH = Path("models") / "mlflow_model_uri.txt"

logger.add("logs/api.log", rotation="1 week", level="INFO")

app = FastAPI(
    title="CSV ML API (MLflow file://)",
    description="API pour entraîner/prédire à partir de CSV en loggant dans MLflow (file backend).",
    version="1.0.0",
)

_model = None
_model_version = "none"  # run_id du modèle courant


# -------------------------------------------------------------------
# Utilitaires
# -------------------------------------------------------------------

def _check_columns_train(df: pd.DataFrame):
    """
    Vérifie que le CSV de TRAIN contient toutes les colonnes d'entrée
    + la colonne cible.
    """
    missing_inputs = [c for c in INPUT_COLUMNS if c not in df.columns]
    missing_target = [] if TARGET_COLUMN in df.columns else [TARGET_COLUMN]
    missing = missing_inputs + missing_target

    if missing:
        raise ValueError(
            f"Colonnes manquantes pour l'entraînement : {missing}. "
            f"Attendu: features={INPUT_COLUMNS} + target={TARGET_COLUMN}"
        )


def _check_columns_predict(df: pd.DataFrame):
    """
    Vérifie que le CSV de PREDICTION contient au moins toutes les colonnes d'entrée.
    La colonne cible peut être absente (cas normal).
    """
    missing_inputs = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing_inputs:
        raise ValueError(
            f"Colonnes manquantes pour la prédiction : {missing_inputs}. "
            f"Attendu au minimum: {INPUT_COLUMNS}"
        )



def _load_model_from_mlflow_uri():
    """
    Tente de charger un modèle MLflow si un URI est stocké dans models/mlflow_model_uri.txt.
    Si le run n'existe pas (nouveau volume mlruns en Docker, etc.), on loggue un warning
    mais on ne fait PAS planter l'API.
    """
    global _model, _model_version

    if not os.path.exists(MLFLOW_MODEL_URI_FILE):
        logger.warning("Aucun fichier mlflow_model_uri.txt trouvé, pas de modèle à charger.")
        return

    with open(MLFLOW_MODEL_URI_FILE, "r", encoding="utf-8") as f:
        model_uri = f.read().strip()

    if not model_uri:
        logger.warning("Fichier mlflow_model_uri.txt vide, pas de modèle à charger.")
        return

    logger.info(f"Tentative de chargement du modèle MLflow : {model_uri}")
    try:
        _model = mlflow.sklearn.load_model(model_uri)
        _model_version = "mlflow_loaded"
        logger.info(f"Modèle MLflow chargé avec succès depuis {model_uri}")
    except Exception as e:
        logger.error(f"Impossible de charger le modèle MLflow depuis {model_uri} : {e}")
        logger.warning("L'API démarre SANS modèle. Il faudra appeler /train pour en créer un.")
        _model = None
        _model_version = None


def _save_model_uri(model_uri: str):
    """Sauvegarde l’URI MLflow du modèle courant dans un fichier local."""
    MLFLOW_URI_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLFLOW_URI_PATH.write_text(model_uri)
    logger.info(f"URI du modèle MLflow sauvegardée dans {MLFLOW_URI_PATH}")


# Chargement éventuel d’un modèle existant au démarrage
_load_model_from_mlflow_uri()


# Routes


@app.get("/health")
def health():
    status = "ready" if _model is not None else "no_model"
    return {"status": status, "model_version": _model_version}


@app.post("/train")
async def train_from_csv(file: UploadFile = File(...)):
    """
    Entraîne (ou réentraîne) un modèle à partir d'un CSV :
    - vérifie la structure du CSV
    - entraîne un NN via ton pipeline
    - logue params + métriques + CSV + modèle dans MLflow (file://)
    - recharge le modèle en mémoire depuis MLflow
    """
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        _check_columns_train(df)

        # Prétraitement + split
        y = df[TARGET_COLUMN].values
        X_raw = df.drop(columns=["montant_pret"])

        # 2) Transformation avec le préprocesseur entraîné
        X = preprocessor.transform(X_raw)

        # 3) Split train / test (tu peux réutiliser ta fonction split si tu veux)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"TRAIN X.shape = {X.shape}, X_train.shape = {X_train.shape}")

        epochs = 60
        batch_size = 32

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Début du run MLflow : {run_id}")

            # Params
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("source", "api/train_from_csv")

            # Entraînement
            model = create_nn_model(X_train.shape[1])
            model, hist = train_model(
                model,
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=epochs,
                batch_size=batch_size,
            )

            # Évaluation
            preds = model_predict(model, X_test)
            perf = evaluate_performance(y_test, preds)
            mlflow.log_metric("MAE", perf["MAE"])
            mlflow.log_metric("MSE", perf["MSE"])
            mlflow.log_metric("R2", perf["R²"])

            # CSV en artifact
            with NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(contents)
                tmp.flush()
                mlflow.log_artifact(tmp.name, artifact_path="training_data")

            # Modèle
            mlflow.sklearn.log_model(model, artifact_path="mlflow_model")
            model_uri = f"runs:/{run_id}/mlflow_model"

        # Rechargement du modèle
        logger.info(f"Fin du run MLflow {run_id}, chargement du modèle : {model_uri}")
        global _model, _model_version
        _model = mlflow.sklearn.load_model(model_uri)
        _model_version = run_id
        _save_model_uri(model_uri)

        return JSONResponse(
            {
                "message": "Modèle entraîné, loggé dans MLflow et chargé pour l'API.",
                "metrics": perf,
                "model_version": _model_version,
                "mlflow_model_uri": model_uri,
                "mlflow_run_id": run_id,
            }
        )

    except ValueError as ve:
        logger.warning(f"Erreur de validation du CSV : {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Erreur pendant l'entraînement avec MLflow.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Prédit à partir d'un CSV (même structure que template.csv).
    """
    if _model is None:
        raise HTTPException(status_code=400, detail="Aucun modèle disponible. Appelle /train avant /predict.")

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        _check_columns_predict(df)

        if "montant_pret" in df.columns:
            X_raw = df.drop(columns=["montant_pret"])
        else:
            X_raw = df.copy()

        X = preprocessor.transform(X_raw)
        logger.info(f"PREDICT X.shape = {X.shape}, model input_shape = {_model.input_shape}")

        # Sécurité : vérifier la compatibilité
        if X.shape[1] != _model.input_shape[1]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Nombre de features incorrect : {X.shape[1]} (préprocessing) "
                    f"vs {int(_model.input_shape[1])} attendus par le modèle. "
                    "Assure-toi que le préprocesseur utilisé est bien celui de l'entraînement."
                ),
            )

        preds = model_predict(_model, X)
        preds = np.asarray(preds).reshape(-1).tolist()

        logger.info(f"Prédiction sur {len(df)} lignes avec modèle version={_model_version}")

        return {
            "n_rows": len(df),
            "predictions": preds,
            "model_version": _model_version,
        }

    except ValueError as ve:
        logger.warning(f"Erreur de validation du CSV : {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Erreur pendant la prédiction.")
        raise HTTPException(status_code=500, detail=str(e))
