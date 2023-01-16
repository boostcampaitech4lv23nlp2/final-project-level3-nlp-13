import io
import json
import os
from pathlib import Path

import requests
import streamlit as st
from app.confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

# st.set_page_config(layout="wide")

# root_password = "a"


def main():
    st.title("Mission Model")
    text = st.text_input("텍스트를 입력해주세요")

    if text:
        st.text(text)
        st.write("Generating...")

        files = {"sentence": text}

        response = requests.post("http://0.0.0.0:30001/input", data=json.dumps(files))
        st.write(f"user : {text}")
        st.write(f"bot : {response.json()}")


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
