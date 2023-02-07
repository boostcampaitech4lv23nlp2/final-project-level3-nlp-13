import argparse
import sys

from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel

sys.path.append("..")  # Adds higher directory to python modules path.
from datetime import datetime

from chatbot.generator.util import Generator
from chatbot.pipeline.data_pipeline import DataPipeline
from chatbot.retriever.elastic_retriever import ElasticRetriever
from classes import UserTweet
from omegaconf import OmegaConf
from pytz import timezone
from spam_filter.spam_filter import SpamFilter
import re

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="base_config")
args, _ = parser.parse_known_args()
config = OmegaConf.load(f"config/{args.config}.yaml")

app = FastAPI()


class User_input(BaseModel):
    sentence: str
    max_len: int
    top_k: int
    top_p: float


# fmt: off
special_tokens = ["BTS", "bts", "RM", "rm", "진", "김석진", "석진", "김남준", "남준", "슈가", "민윤기", "윤기", "제이홉", "정호석", "지민", "박지민", "뷔", "김태형", "태형", "V", "정국", "전정국", "아미", "빅히트", "하이브", "아미", "보라해" ] #TO-Do
# fmt: on

today = datetime.now(timezone("Asia/Seoul")).strftime("%m%d")
generator = Generator(config)

def islanguage(text):
    language = re.compile('[ㄱ-ㅣ가-힣a-zA-Z]')
    if language.search(text):
        return True
    return False

@app.post("/input", description="주문을 요청합니다")
async def make_chat(data: User_input):
    text = data.dict()["sentence"]
    max_len = data.dict()["max_len"]
    top_k = data.dict()["top_k"]
    top_p = data.dict()["top_p"]

    is_spam = SpamFilter().sentences_predict(text)  # 1이면 스팸, 0이면 아님
    if not text.strip():
        return "입력이 없어"
    elif not islanguage(text):
        return "문자를 입력해줘"
    elif is_spam:
        return "글쎄..."
    else:
        # 3-1. 전처리 & 리트리버
        data_pipeline = DataPipeline(log_dir="log", special_tokens=special_tokens)
        elastic_retriever = ElasticRetriever()
        retrieved = elastic_retriever.return_answer(text)
        print("🔥 ", data_pipeline)
        print("🔥🔥 ", text)
        print("🔥🔥🔥 ", retrieved.query)

        if retrieved.query is not None:
            my_answer = data_pipeline.correct_grammar(retrieved)
        else:
            # 3-2. 전처리 없이? 생성모델
            my_answer = generator.get_answer(text, 1, max_len, top_k, top_p)

            if "<account>" in my_answer:
                my_answer = my_answer.replace("<account>", "유저")

    # log: user message + screen name + bot answer
    data_pipeline.log(
        new_entries=[UserTweet(screen_name="익명의 유저", message=text, reply=my_answer)],
        save_name=today,
    )

    # inference_result = generator.get_answer(text, 1, max_len, top_k, top_p)

    return my_answer
