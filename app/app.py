import streamlit as st
import requests
import pandas as pd
from io import StringIO
import os

API_URL_DEFAULT = os.getenv("API_URL_DEFAULT", "http://localhost:8000")

st.set_page_config(page_title="Demo CSV ML", layout="centered")
st.title("Interface CSV – Entraînement & Prédiction")

st.sidebar.header("Configuration API")
api_url = st.sidebar.text_input("URL de l'API", API_URL_DEFAULT)

st.sidebar.markdown("Exemples de routes :")
st.sidebar.code(f"{api_url}/health\n{api_url}/train\n{api_url}/predict")

st.write(
    "Les fichiers CSV doivent respecter la structure du fichier template "
    "(colonnes : nom, prenom, age, taille, poids, sexe, ... , montant_pret)."
)


# Section health


st.subheader("Health check")
if st.button("Tester /health"):
    try:
        resp = requests.get(f"{api_url}/health")
        resp.raise_for_status()
        st.json(resp.json())
    except Exception as e:
        st.error(f"Erreur lors de l'appel à /health : {e}")

st.markdown("---")


# Section entraînement


st.subheader("Entraîner / réentraîner le modèle à partir d'un CSV")

train_file = st.file_uploader(
    "Choisir un CSV pour l'entraînement",
    type=["csv"],
    key="train_csv",
)

if st.button("Lancer l'entraînement") and train_file is not None:
    try:
        # Afficher un aperçu
        df = pd.read_csv(train_file)
        st.write("Aperçu du CSV :")
        st.dataframe(df.head())

        # Repartir depuis le début du buffer pour l'envoi
        train_file.seek(0)
        files = {"file": (train_file.name, train_file.read(), "text/csv")}

        with st.spinner("Entraînement en cours..."):
            resp = requests.post(f"{api_url}/train", files=files)
            resp.raise_for_status()
        data = resp.json()
        st.success("Entraînement terminé ✅")
        st.json(data)   # contiendra metrics + mlflow_run_id + mlflow_model_uri

    except Exception as e:
        st.error(f"Erreur pendant l'entraînement : {e}")

st.markdown("---")


# Section prédiction


st.subheader("Faire des prédictions à partir d'un CSV")

predict_file = st.file_uploader(
    "Choisir un CSV pour la prédiction",
    type=["csv"],
    key="predict_csv",
)

if st.button("Lancer la prédiction") and predict_file is not None:
    try:
        df = pd.read_csv(predict_file)
        st.write("Aperçu du CSV :")
        st.dataframe(df.head())

        predict_file.seek(0)
        files = {"file": (predict_file.name, predict_file.read(), "text/csv")}

        with st.spinner("Prédiction en cours..."):
            resp = requests.post(f"{api_url}/predict", files=files)
            resp.raise_for_status()
        data = resp.json()

        st.success("Prédiction terminée ✅")
        st.write(f"Nombre de lignes : {data['n_rows']}")
        preds = data["predictions"]

        # Afficher les prédictions à côté des entrées
        df_out = df.copy()
        df_out["prediction_montant_pret"] = preds
        st.dataframe(df_out.head())

        # Option : permettre le téléchargement des résultats
        csv_out = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger les résultats en CSV",
            data=csv_out,
            file_name="predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
