import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

path = "/".join(str(Path(__file__)).split("/")[:-1])
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from chatbot.generator.util import Generator
from chatbot.pipeline.data_pipeline import DataPipeline
from chatbot.retriever.elastic_retriever import ElasticRetriever
from database.mongodb import MongoDB
from omegaconf import OmegaConf
from pytz import timezone
from spam_filter.spam_filter import SpamFilter
from twitter.tweet_pipeline import TwitterPipeline
from utils.classes import BotReply, UserTweet

# fmt: off
special_tokens = ["BTS", "bts", "RM", "rm", "진", "김석진", "석진", "김남준", "남준", "슈가", "민윤기", "윤기", "제이홉", "정호석", "지민", "박지민", "뷔", "김태형", "태형", "V", "정국", "전정국", "아미", "빅히트", "하이브", "아미", "보라해" ] #TO-Do
# fmt: on


def main():
    config = OmegaConf.load(f"{path}/utils/base_config.yaml")

    # init modules
    spam_filter = SpamFilter()
    twitter_pipeline = TwitterPipeline(FILE_NAME=f"{path}/twitter/last_seen_id.txt", bot_username="armybot_13")
    data_pipeline = DataPipeline(log_dir="log", special_tokens=special_tokens)
    elastic_retriever = ElasticRetriever()
    generator = Generator(config)
    db = MongoDB()

    today = datetime.now(timezone("Asia/Seoul")).strftime("%m%d")

    # twitter api에서 메시지 불러오기
    new_tweets = twitter_pipeline.get_mentions()
    if len(new_tweets) == 0:
        # 새 메시지가 없으면
        return
    else:
        for tweet in reversed(new_tweets):
            time_log = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
            user_message = tweet.message.lower()

            # 스팸 필터링
            is_spam = spam_filter.sentences_predict(user_message)  # 1이면 스팸, 0이면 아님
            if is_spam:
                my_reply = reply_to_spam = "...."
                twitter_pipeline.reply_tweet(tweet=tweet, reply=reply_to_spam)
                score = 0.0
            else:
                # 리트리버
                retrieved = elastic_retriever.return_answer(user_message)
                if retrieved.query is not None:
                    my_reply = data_pipeline.correct_grammar(retrieved)
                    score = retrieved.bm25_score
                else:
                    # 생성모델
                    my_reply = generator.get_answer(user_message, 1, 256)
                    # 후처리
                    my_reply = data_pipeline.postprocess(my_reply, tweet.user_screen_name)
                    score = 0.0
                # twitter로 보내기
                twitter_pipeline.reply_tweet(tweet=tweet, reply=my_reply)
                # twitter 좋아요
                twitter_pipeline.like_tweet(tweet)

            # logging
            record = BotReply(
                tweet=tweet,
                reply=my_reply,
                score=score,
                is_spam=bool(is_spam),
                time=time_log,
            ).__dict__
            print(record)
            db.insert_one(record)

    print("🔥🔥🔥🔥🔥🔥🔥🔥🔥", config)
    # return main(spam_filter, twitter_pipeline, data_pipeline, elastic_retriever, generator, db)


# with 구문으로 DAG 정의를 시작합니다.
with DAG(
    dag_id="Armybot13",  # DAG의 식별자용 아이디입니다.
    description="Run Agent.py",  # DAG에 대해 설명합니다.
    start_date=datetime(2023, 2, 17),  # 시작 날짜
    schedule_interval="*/30 * * * *",  # 30분마다 실행합니다.
    tags=["my_dags"],  # 태그 목록을 정의합니다. 추후에 DAG을 검색하는데 용이합니다.
) as dag:

    # 테스크를 정의합니다.
    # python 함수인 main를 실행합니다.
    t1 = PythonOperator(
        task_id="main",
        python_callable=main,
        depends_on_past=True,
        owner="junnyeong",  # 이 작업의 오너입니다. 보통 작업을 담당하는 사람 이름을 넣습니다.
        retries=1,  # 이 테스크가 실패한 경우, 1번 재시도 합니다.
        retry_delay=timedelta(minutes=1),  # 재시도하는 시간 간격은 1분입니다.
    )

    t1
