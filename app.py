import streamlit as st
import numpy as np
import pickle

# ------------------ CONFIG PAGE ------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ------------------ LOAD MODELS ------------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

models = {
    "Random Forest": rf_model,
    "Support Vector Machine (SVM)": svm_model,
    "K-Nearest Neighbors (KNN)": knn_model
}

# ------------------ TITLE ------------------
st.markdown(
    "<h1 style='text-align:center;'>🚢 Titanic Survival Prediction</h1>",
    unsafe_allow_html=True
)
st.caption("Comparaison de plusieurs modèles de Machine Learning")

st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.header(" Choix du modèle")
model_name = st.sidebar.selectbox(
    "Sélectionner le modèle",
    list(models.keys())
)

model = models[model_name]

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

    if prediction[0] == 1:
        st.success(f"✅ Survie prédite avec le modèle **{model_name}**")
    else:
        st.error(f"❌ Non-survie prédite avec le modèle **{model_name}**")


# ------------------ INFO ------------------
st.divider()
st.markdown(
    f"""
    ### ℹ️ Informations
    - Modèle sélectionné : **{model_name}**
    - Données : Titanic Dataset
    - Prétraitement : nettoyage, encodage, standardisation
    """
)
