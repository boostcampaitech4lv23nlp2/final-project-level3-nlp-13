from argparse import ArgumentParser
from datetime import datetime

from chatbot.generator.util import Generator
from chatbot.pipeline.data_pipeline import DataPipeline
from chatbot.retriever.elastic_retriever import ElasticRetriever
from classes import UserTweet
from omegaconf import OmegaConf
from pytz import timezone
from spam_filter.spam_filter import SpamFilter
from twitter.data_pipeline import TwitterPipeline, TwitterupdatePipeline

# fmt: off
special_tokens = ["BTS", "bts", "RM", "rm", "진", "김석진", "석진", "김남준", "남준", "슈가", "민윤기", "윤기", "제이홉", "정호석", "지민", "박지민", "뷔", "김태형", "태형", "V", "정국", "전정국", "아미", "빅히트", "하이브", "아미", "보라해" ] #TO-Do
# fmt: on


def main(config):
    today = datetime.now(timezone("Asia/Seoul")).strftime("%m%d")

    # 1. twitter api에서 메시지 불러오기
    last_seen_id, user_name, tweet = TwitterPipeline(FILE_NAME="./twitter/last_seen_id.txt", username="@wjlee_nlp").reply_to_tweets()

    # 2. 스팸 필터링
    is_spam = SpamFilter().sentences_predict(tweet)  # 1이면 스팸, 0이면 아님
    if is_spam:
        TwitterupdatePipeline(username=user_name, output_text="글쎄...", last_seen_id=last_seen_id).update()

    else:
        # 3-1. 전처리 & 리트리버
        data_pipeline = DataPipeline(log_dir="log", special_tokens=special_tokens)
        # data_pipeline.log(new_entries=[tweet], save_name=today)
        elastic_retriever = ElasticRetriever()
        retrieved = elastic_retriever.return_answer(tweet)
        if retrieved.bm25_score is not None:
            my_answer = data_pipeline.correct_grammar(retrieved)
        else:
            # 3-2. 전처리 없이? 생성모델
            generator = Generator(config)
            my_answer = generator.get_answer(tweet, 1, 256)

            # TO-DO: 생성 결과후처리

        # 6. twitter로 보내기
        TwitterupdatePipeline(username=user_name, output_text=my_answer, last_seen_id=last_seen_id).update()


if __name__ == "__main__":

    parser = ArgumentParser()  # HfArgumentParser((AgentArguments))
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--query", type=str)
    parser.add_argument("--config", "-c", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./config/{args.config}.yaml")

    # TO-DO: 각 submodule init은 여기서 하고 instances를 main안에 넣어주기
    main(config)
