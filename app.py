import streamlit as st
import numpy as np
import pickle

# ------------------ CONFIG PAGE ------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------ TITLE ------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🚢 Titanic Survival Prediction</h1>
    <p style='text-align: center; font-size:18px;'>
    Application de Machine Learning pour prédire la survie des passagers du Titanic
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.header("🧾 Informations du passager")

pclass = st.sidebar.selectbox("Classe du passager", [1, 2, 3])
sex = st.sidebar.selectbox("Sexe", ["male", "female"])
age = st.sidebar.slider("Âge", 1, 80, 30)
sibsp = st.sidebar.slider("Frères/Sœurs / Conjoint", 0, 8, 0)
parch = st.sidebar.slider("Parents / Enfants", 0, 6, 0)
fare = st.sidebar.slider("Prix du billet", 0.0, 600.0, 50.0)
embarked = st.sidebar.selectbox("Port d’embarquement", ["S", "C", "Q"])

# ------------------ ENCODING ------------------
sex = 1 if sex == "male" else 0
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

X_user = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# ------------------ PREDICTION ------------------
st.subheader("🔍 Résultat de la prédiction")

if st.button("Prédire la survie"):
    X_scaled = scaler.transform(X_user)
    prediction = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    survival_prob = proba[0][1] * 100

    if prediction[0] == 1:
        st.success(f"✅ Le passager a de fortes chances de **SURVIVRE** ({survival_prob:.2f}%)")
    else:
        st.error(f"❌ Le passager a de faibles chances de **SURVIVRE** ({survival_prob:.2f}%)")

# ------------------ INFO SECTION ------------------
st.divider()

st.markdown(
    """
    ### ℹ️ À propos du modèle
    - Modèle utilisé : **Random Forest**
    - Données : **Titanic Dataset**
    - Prétraitement : nettoyage, encodage, standardisation
    - Objectif : prédire la survie d’un passager
    """
)

st.caption("Projet Machine Learning – Titanic | ENSA")
