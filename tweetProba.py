from joblib import dump, load
import streamlit as st
clf = load('filename.joblib') 

text = st.text_input("Ecrire un tweet :")

st.write("Prediction 0 - hate speech / 1 - offensive language / 2 - neither")
st.write(clf.predict([text]))

st.write("Predict Proba")
st.write(clf.predict_proba([text]))
