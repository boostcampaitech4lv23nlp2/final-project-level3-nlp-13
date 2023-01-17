import datetime
import json
import logging
import os
import sys

import pytz
import requests
import streamlit as st
from streamlit_chat import message


class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""

    def converter(self, timestamp):
        # Create datetime in UTC
        dt = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        # Change datetime's timezone
        return dt.astimezone(pytz.timezone("Asia/Seoul"))  # 한국 시간으로 timezone 변경

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="seconds")
            except TypeError:
                s = dt.isoformat(" ")
        return s


def CreateLogger(logger_name):
    # logging 설정
    logger = logging.getLogger(logger_name)

    # 중복 logging이 되지 않도록 logger가 존재하면 새로운 logger가 생성되지 않도록 기존 logger 반환
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger  # Logger already exists

    if not os.path.exists("./logs"):
        os.makedirs("./logs")
        with open("./logs/service.log", "w+") as f:
            f.write("Time\tUser\tBot\n")
    LOG_FORMAT = "%(asctime)s\t%(message)s"
    file_handler = logging.FileHandler("./logs/service.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


logger = CreateLogger("ChatbotLogger")


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

        logger.info(f"{user_input}\t{output.strip()}")

    for i in range(len(st.session_state["past"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        if len(st.session_state["generated"]) > i:
            message(st.session_state["generated"][i], key=str(i) + "_bot")


main()
