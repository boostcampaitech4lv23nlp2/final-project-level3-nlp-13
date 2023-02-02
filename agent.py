import os
from argparse import ArgumentParser

# from omegaconf import OmegaConf
from datetime import datetime

import numpy as np
import pandas as pd

# from chatbot.retriever.elastic_retriever import ElasticRetriever
from chatbot.pipeline.data_pipeline import DataPipeline
from classes import UserTweet
from pytz import timezone

# fmt: off
special_tokens = ["BTS", "bts", "RM", "rm", "진", "김석진", "석진", "김남준", "남준", "슈가", "민윤기", "윤기", "제이홉", "정호석", "지민", "박지민", "뷔", "김태형", "태형", "V", "정국", "전정국", "아미", "빅히트", "하이브", "아미", "보라해"] #TO-Do
# fmt: on


def main():
    today = datetime.now(timezone("Asia/Seoul")).strftime("%m%d")

    # 1. twitter api에서 메시지 불러오기
    tweet = UserTweet(user_id="34k73", screen_name="아미1", message="전정국 최고")

    # 2. 스팸 필터링

    # 3-1. 전처리 & 리트리버

    data_pipeline = DataPipeline(log_dir="log", special_tokens=special_tokens)
    data_pipeline.log(new_entries=[tweet], save_name=today)
    # elastic_retriever = ElasticRetriever()

    # 3-2. 전처리 없이? 생성모델

    # 4. 리트리버 결과와 생성 결과 비교 및 선택, 후처리

    # 5. 스팸 필터링 (욕설 제거 등)

    # 6. twitter로 보내기


if __name__ == "__main__":

    parser = ArgumentParser()  # HfArgumentParser((AgentArguments))
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--query", type=str)
    parser.add_argument("--config", "-c", type=str, default="retriever_config")

    args, _ = parser.parse_known_args()
    # config = OmegaConf.load(f"./chatbot/retriever/{args.config}.yaml")
    main()
