import streamlit as st
from api_test_openai import main
import gc

gc.enable()
st.set_page_config(page_title="zeilertech", page_icon=":scales:")

st.title("LawGPT")
question = st.text_input("Gib deine rechtliche Frage hier ein")

if st.button("Submit"):
    with st.spinner('Konzipienten werden gepeitscht...'):
        try:
            response = main(question)
        except Exception as e:
            st.error(f"Konzipient braucht mokka latte mit pumpkin spice (no gluten) weil {str(e)}")
        else:
            st.text_area("Rechtsgutachten:", response)