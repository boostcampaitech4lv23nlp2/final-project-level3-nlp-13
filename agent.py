from argparse import ArgumentParser
from datetime import datetime

from chatbot.generator.util import Generator
from chatbot.pipeline.data_pipeline import DataPipeline
from chatbot.retriever.elastic_retriever import ElasticRetriever
from twitter.tweet_pipeline import TwitterPipeline
from classes import UserTweet
from omegaconf import OmegaConf
from pytz import timezone
from spam_filter.spam_filter import SpamFilter

# fmt: off
special_tokens = ["BTS", "bts", "RM", "rm", "진", "김석진", "석진", "김남준", "남준", "슈가", "민윤기", "윤기", "제이홉", "정호석", "지민", "박지민", "뷔", "김태형", "태형", "V", "정국", "전정국", "아미", "빅히트", "하이브", "아미", "보라해" ] #TO-Do
# fmt: on


def main(spam_filter, twitter_pipeline, data_pipeline, elastic_retriever, generator):
    today = datetime.now(timezone("Asia/Seoul")).strftime("%m%d")

    # 1. twitter api에서 메시지 불러오기
    new_tweets = twitter_pipeline.get_mentions()
    print(new_tweets)
    for tweet in reversed(new_tweets):
        print(tweet)
        usr_msg = tweet.text

        # 2. 스팸 필터링
        is_spam = spam_filter.sentences_predict(usr_msg)  # 1이면 스팸, 0이면 아님
        if is_spam:
            reply_to_spam = "닥쳐 말포이"
            twitter_pipeline.reply_tweet(tweet=tweet, reply=reply_to_spam)
        else:
            # 3-1. 전처리 & 리트리버
            # usr_msg_preprocessed = data_pipeline.preprocess(usr_msg)
            # print(usr_msg_preprocessed)
            retrieved = elastic_retriever.return_answer(usr_msg)
            if retrieved.query is not None:
                my_reply = data_pipeline.correct_grammar(retrieved)
            else:
                # 3-2. 전처리 없이? 생성모델
                my_reply = generator.get_answer(usr_msg, 1, 256)

                # TO-DO: 생성 결과후처리

            # 6. twitter로 보내기
            twitter_pipeline.reply_tweet(tweet=tweet, reply=my_reply)


if __name__ == "__main__":

    parser = ArgumentParser()  # HfArgumentParser((AgentArguments))
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--query", type=str)
    parser.add_argument("--config", "-c", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./config/{args.config}.yaml")

    # init modules
    spam_filter = SpamFilter()
    twitter_pipeline = TwitterPipeline(FILE_NAME="./twitter/last_seen_id.txt", username="@wjlee_nlp")
    data_pipeline = DataPipeline(log_dir="log", special_tokens=special_tokens)
    elastic_retriever = ElasticRetriever()
    generator = Generator(config)

    main(spam_filter, twitter_pipeline, data_pipeline, elastic_retriever, generator)
