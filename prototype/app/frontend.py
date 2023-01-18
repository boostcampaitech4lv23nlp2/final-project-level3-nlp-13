import json

import requests
import streamlit as st
from streamlit_chat import message


def main():
    st.title("nlpotato chatbot")
    st.sidebar.subheader("Generation Settings")
    max_len = st.sidebar.slider("max length", 30, 100, value=60)
    top_k = st.sidebar.slider("top k sampling", 10, 50, value=25)
    top_p = st.sidebar.slider("top p sampling", 0.0, 1.0, step=0.01, value=0.95)

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    with st.form("form", clear_on_submit=True):
        user_input = st.text_input("User: ", "")
        submitted = st.form_submit_button("전송")

    if submitted and user_input:
        if user_input:
            files = {"sentence": user_input, "max_len": max_len, "top_k": top_k, "top_p": top_p}

            response = requests.post("http://0.0.0.0:30001/input", data=json.dumps(files))
            output = response.json()

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    for i in range(len(st.session_state["past"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        if len(st.session_state["generated"]) > i:
            message(st.session_state["generated"][i], key=str(i) + "_bot")


main()
