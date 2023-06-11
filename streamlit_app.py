import streamlit as st
from api_test_openai import main

st.title("Legal Chatbot")
question = st.text_input("Please enter your legal question here:")

if st.button("Submit"):
    response = main(question)  # Assuming your main function accepts the question as an argument and returns the response
    st.markdown(f"**Response:**\n{response}")