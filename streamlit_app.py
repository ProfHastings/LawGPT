import streamlit as st
from api_test_openai import main

st.set_page_config(page_title="zeilertech", page_icon=":scales:")

st.title("Legal Chatbot")
question = st.text_input("Please enter your legal question here:")

if st.button("Submit"):
    with st.spinner('Konzipienten werden gepeitscht...'):
        try:
            response = main(question)
        except Exception as e:
            st.error(f"Konzipient braucht mokka latte mit pumpkin spice (no gluten) weil {str(e)}")
        else:
            st.text_area("Rechtsgutachten:", response)