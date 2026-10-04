"""
App Streamlit — Prédiction de l'état d'un téléphone d'occasion
(D'occasion / Neuf / Réconditionné / Venant)

Reprend le pipeline du TP3 : encodage des variables catégorielles (LabelEncoder),
normalisation (StandardScaler), puis Gradient Boosting (meilleur modèle retenu
après comparaison de 7 modèles).

Fichiers nécessaires dans le même dossier :
- app.py
- gb_model.joblib
- encoders.joblib   (LabelEncoder pour : adresse, marque, etat — dans cet ordre)
- scaler.joblib
- uniques.joblib    (valeurs possibles de : adresse, marque, etat — dans cet ordre)
- model_info.joblib
- requirements.txt
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="État d'un téléphone d'occasion", page_icon="📱")


@st.cache_resource
def load_artifacts():
    model = joblib.load("gb_model.joblib")
    encoders = joblib.load("encoders.joblib")  # [adresse, marque, etat]
    scaler = joblib.load("scaler.joblib")
    uniques = joblib.load("uniques.joblib")    # [adresse, marque, etat]
    info = joblib.load("model_info.joblib")
    return model, encoders, scaler, uniques, info


model, encoders, scaler, uniques, info = load_artifacts()
enc_adresse, enc_marque, enc_etat = encoders
adresses, marques, classes_etat = uniques
feature_cols = info["feature_cols"]  # ['prix', 'adresse', 'marque', 'dim_ecr', 'ram', 'stockage']

st.title("📱 Prédiction de l'état d'un téléphone d'occasion")
st.write(
    f"Modèle utilisé : **{info['best_model_name']}** "
    f"(F1 validation = {info['results'].iloc[0]['F1']:.3f})"
)


def predict_one(prix, adresse, marque, dim_ecr, ram, stockage):
    adresse_enc = enc_adresse.transform([adresse])[0]
    marque_enc = enc_marque.transform([marque])[0]
    x_new = np.array([[prix, adresse_enc, marque_enc, dim_ecr, ram, stockage]], dtype=float)
    x_new = scaler.transform(x_new)
    pred = model.predict(x_new)[0]
    return enc_etat.inverse_transform([pred])[0]


tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

with tab1:
    col1, col2 = st.columns(2)
    prix = col1.number_input("Prix (FCFA)", min_value=0, value=100000, step=1000)
    dim_ecr = col2.number_input("Dimension écran (pouces)", min_value=0.0, value=6.1, step=0.1)

    col3, col4 = st.columns(2)
    adresse = col3.selectbox("Adresse / Quartier", adresses)
    marque = col4.selectbox("Marque", marques)

    col5, col6 = st.columns(2)
    ram = col5.number_input("RAM (Go)", min_value=0, value=4, step=1)
    stockage = col6.number_input("Stockage (Go)", min_value=0, value=64, step=1)

    if st.button("Prédire l'état", type="primary"):
        etat = predict_one(prix, adresse, marque, dim_ecr, ram, stockage)
        st.success(f"État prédit : **{etat}**")

with tab2:
    st.write(
        "Importer un fichier CSV avec les colonnes : "
        "`prix, adresse, marque, dim_ecr, ram, stockage`"
    )
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df_in = pd.read_csv(uploaded_file)
        predictions = [
            predict_one(row["prix"], row["adresse"], row["marque"], row["dim_ecr"], row["ram"], row["stockage"])
            for _, row in df_in.iterrows()
        ]
        df_in["etat_predit"] = predictions
        st.dataframe(df_in)
        st.download_button(
            "Télécharger les prédictions (CSV)",
            df_in.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv",
        )
