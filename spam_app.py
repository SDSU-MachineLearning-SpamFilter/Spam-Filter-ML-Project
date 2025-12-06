import streamlit as st
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("pac_model.pkl") ## load models here with pkl generated from script
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

pac_model, vectorizer = load_model()

st.title(" Email Spam Filter")
st.write(
    "Paste an email below and the model will classify it as **Spam** or **Ham (Not Spam)**. "
    "This demo uses several algorithms trained on the Enron dataset."
)

email_text = st.text_area("Email text:", height=200)

if st.button("Classify"):
    if not email_text.strip():
        st.warning("Please enter some email text first.")
    else:
        X = vectorizer.transform([email_text])
        pred = pac_model.predict(X)[0]
        score = pac_model.decision_function(X)[0]

        if pred.lower() == "spam":
            st.error(f" Prediction: **SPAM**")
        else:
            st.success(f" Prediction: **HAM (Not Spam)**")

        st.caption(f"Decision score: {score:.3f} (higher = more strongly classified as spam)")
