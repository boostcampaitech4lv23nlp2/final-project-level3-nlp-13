import argparse
import sys

from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel

sys.path.append("..")  # Adds higher directory to python modules path.
import re
from datetime import datetime

from chatbot.generator.util import Generator
from chatbot.pipeline.data_pipeline import DataPipeline
from chatbot.retriever.elastic_retriever import ElasticRetriever

# from database.mongodb import MongoDB
from omegaconf import OmegaConf
from pytz import timezone
from spam_filter.spam_filter import SpamFilter
from utils.classes import BotReply, UserTweet

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
spam_filter = SpamFilter()
data_pipeline = DataPipeline(log_dir="log", special_tokens=special_tokens)
elastic_retriever = ElasticRetriever()
generator = Generator(config)
# db = MongoDB()


@app.post("/input", description="주문을 요청합니다")
async def make_chat(data: User_input):
    time_log = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    user_message = data.dict()["sentence"].lower()
    max_len = data.dict()["max_len"]
    top_k = data.dict()["top_k"]
    top_p = data.dict()["top_p"]

    # 스팸 필터링
    is_spam = spam_filter.sentences_predict(user_message)  # 1이면 스팸, 0이면 아님
    if is_spam:
        return "...."
    else:
        # 리트리버
        retrieved = elastic_retriever.return_answer(user_message)
        if retrieved.query is not None:
            my_reply = data_pipeline.correct_grammar(retrieved)
            score = retrieved.bm25_score
        else:
            # 생성모델
            my_reply = generator.get_answer(user_message, 1, max_len, top_k, top_p)
            # 후처리
            my_reply = data_pipeline.postprocess(my_reply, "유저")
            score = 0.0

    # logging
    record = BotReply(
        tweet=user_message,
        reply=my_reply,
        score=score,
        is_spam=bool(is_spam),
        time=time_log,
    ).__dict__
    print(record)
    # db.insert_one(record)

    return my_reply
