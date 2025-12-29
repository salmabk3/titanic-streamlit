import streamlit as st
import numpy as np
import pickle

# Configuration de la page
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Chargement du modèle et du scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Titre
st.title("🚢 Titanic Survival Prediction")
st.markdown(
    """
    Cette application permet de prédire la **survie d’un passager du Titanic**
    à partir de ses caractéristiques personnelles et économiques.
    """
)

st.divider()

# Interface utilisateur
pclass = st.selectbox("Classe du passager", [1, 2, 3])
sex = st.selectbox("Sexe", ["male", "female"])
age = st.slider("Âge", 1, 80, 30)
sibsp = st.slider("Nombre de frères/sœurs / conjoint", 0, 8, 0)
parch = st.slider("Nombre de parents / enfants", 0, 6, 0)
fare = st.slider("Prix du billet", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port d’embarquement", ["S", "C", "Q"])

# Encodage manuel (IDENTIQUE à ton notebook)
sex = 1 if sex == "male" else 0
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

# Données utilisateur
X_user = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Prédiction
if st.button("🔍 Prédire la survie"):
    X_user_scaled = scaler.transform(X_user)
    prediction = model.predict(X_user_scaled)

    if prediction[0] == 1:
        st.success("✅ Le passager a de fortes chances de **SURVIVRE**")
    else:
        st.error("❌ Le passager a de faibles chances de **SURVIVRE**")

st.divider()
st.caption("Projet Machine Learning — Dataset Titanic")

