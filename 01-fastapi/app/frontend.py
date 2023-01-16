import io
import json
import os
from pathlib import Path

import requests
import streamlit as st
from app.confirm_button_hack import cache_on_button_press
from streamlit_chat import message

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

# st.set_page_config(layout="wide")

# root_password = "a"


def main():
    st.title("Mission Model")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    with st.form("form", clear_on_submit=True):
        user_input = st.text_input("User: ", "")
        submitted = st.form_submit_button("전송")

    if submitted and user_input:
        if user_input:
            files = {"sentence": user_input}

            response = requests.post("http://0.0.0.0:30001/input", data=json.dumps(files))
            output = response.json()

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    for i in range(len(st.session_state["past"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        if len(st.session_state["generated"]) > i:
            message(st.session_state["generated"][i], key=str(i) + "_bot")


# @cache_on_button_press("Authenticate")
# def authenticate(password) -> bool:
#     return password == root_password


# password = st.text_input("password", type="password")

# if authenticate(password):
#     st.success("You are authenticated!")
#     main()
# else:
#     st.error("The password is invalid.")
main()
