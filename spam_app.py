import streamlit as st
import joblib

## load models and vectorizer
@st.cache_resource
def load_all_models():
    models = {
        "Passive-Aggressive": joblib.load("pac_model.pkl")
        # insert more models here after generating your pkl file
    }
    vectorizer = joblib.load("pac_vectorizer.pkl")
    return models, vectorizer

models, vectorizer = load_all_models()

## prompt for user input
st.title(" Email Spam Filter")
st.write(
    "Paste an email below and the model will classify it as **Spam** or **Ham (Not Spam)**. "
    "This demo uses several algorithms trained on the Enron dataset."
)

## drop down menu
model_name = st.selectbox(
    "Choose classifier:",
    list(models.keys()),
    index=0  # default selection
)

st.write(f"Currently using: **{model_name}**")

## area to insert email 
email_text = st.text_area("Email text:", height=200)

## classify spam or ham 
if st.button("Classify"):
    if not email_text.strip():
        st.warning("Please enter some email text first.")
    else:
        model = models[model_name]  # grab the model name
        X = vectorizer.transform([email_text]) # vectorize the input
        pred = model.predict(X)[0] # run the pred
        score = model.decision_function(X)[0] # return decision score

        if pred.lower() == "spam":
            st.error(f" Prediction: **SPAM**")
        else:
            st.success(f" Prediction: **HAM (Not Spam)**")

        st.caption(f"Decision score: {score:.3f} (higher = more strongly classified as spam)")
