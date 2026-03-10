import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.title("RAG AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a question")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    response = requests.post(API_URL, json={"query": user_input})
    bot_reply = response.json()["response"]

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})


for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])