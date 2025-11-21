from io import StringIO

import pandas as pd
from fastapi.testclient import TestClient
import mlflow

from api.main import app, INPUT_COLUMNS, TARGET_COLUMN


client = TestClient(app)


def _make_csv(include_target: bool = True, n_rows: int = 10) -> StringIO:
    """
    Construit un CSV de test avec n_rows lignes valide pour l'API.
    Si include_target=True, génère également la colonne cible.
    """

    base = {
        "prenom": ["Zoé", "Margot", "Arthur"],
        "age": [30, 40, 50],
        "taille": [165.0, 170.0, 175.0],
        "poids": [60.0, 70.0, 80.0],
        "sexe": ["F", "H", "H"],
        "sport_licence": ["oui", "oui", "oui"],
        "niveau_etude": ["bac+2", "bac+2", "master"],
        "region": ["Normandie", "Occitanie", "Bretagne"],
        "smoker": ["non", "non", "non"],
        "nationalité_francaise": ["oui", "oui", "oui"],
        "revenu_estime_mois": [2000, 2500, 3000],
    }

    # Étendre chaque colonne pour obtenir n_rows lignes
    data = {
        k: (v * ((n_rows // len(v)) + 1))[:n_rows]
        for k, v in base.items()
    }

    # Créer le DataFrame avec les colonnes d’entrée
    df = pd.DataFrame(data, columns=INPUT_COLUMNS)

    # Générer la colonne cible correctement (n_rows valeurs)
    if include_target:
        base_target = [5000.0, 10000.0, 15000.0]
        df[TARGET_COLUMN] = (base_target * ((n_rows // len(base_target)) + 1))[:n_rows]

    # Convertir en CSV
    buf = StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf



def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_train_and_predict(tmp_path):
    """
    - configure MLflow sur un backend file:// temporaire
    - crée un experiment dédié
    - appelle /train puis /predict
    """
    # 1) Configurer MLflow pour ce test
    mlruns_dir = tmp_path / "mlruns_test"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"file:{mlruns_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("api_test")  # créé si n'existe pas

    # 2) /train avec un petit CSV
    train_buf = _make_csv(include_target=True)
    files = {"file": ("train.csv", train_buf.getvalue(), "text/csv")}
    resp = client.post("/train", files=files)
    assert resp.status_code == 200, resp.text
    train_data = resp.json()
    assert "metrics" in train_data
    assert "mlflow_run_id" in train_data

    # 3) /predict avec un CSV sans cible
    predict_buf = _make_csv(include_target=False)
    files = {"file": ("predict.csv", predict_buf.getvalue(), "text/csv")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200, resp.text
    pred_data = resp.json()
    assert "predictions" in pred_data
    assert len(pred_data["predictions"]) == 10
