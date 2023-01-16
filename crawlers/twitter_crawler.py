import datetime
import glob
import os
import pickle
import re
from collections import defaultdict

import pandas as pd
import pytz
import tweepy
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
TWITTER_CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET_KEY = os.environ.get("TWITTER_CONSUMER_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET_TOKEN = os.environ.get("TWITTER_ACCESS_SECRET_TOKEN")


class TwitterCrawler:
    def __init__(self):
        """
        Args:
            screen_name (str): 크롤링할 특정 트위터 유저 screen_name
        """
        self.auth = tweepy.OAuthHandler(
            TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET_KEY
        )
        self.auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET_TOKEN)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)

        self.save_path = "data/raw_data/twitter"

        self.screen_name = None  # 호출 시 입력될 특정 트위터 유저 screen_name
        self.screen_names = None  # get_following_screen_names() 실행을 통해 저장될 변수

    def get_following_screen_names(self, screen_name):
        """특정 트위터 계정이 팔로우 하는 유저들의 screen_name list를 반환
        Args:
            screen_name (str): 크롤링할 특정 트위터 유저 screen_name
        Returns:
            following_screen_names (list): 해당 유저가 follow하는 유저들의 screen_name list
        """
        print(f"***** {self.screen_name}의 following 목록을 가져오는 중입니다... ******")
        following_screen_names = []

        get_followed_ids = self.api.get_friend_ids(screen_name=self.screen_name)
        for followed_id in get_followed_ids:
            following_screen_names.append(
                self.api.get_user(user_id=followed_id).screen_name
            )
        following_screen_names.append(self.screen_name)
        if len(following_screen_names) > 100:
            following_screen_names = following_screen_names[:100]

        print(f"***** following 목록을 가져오기 완료 : {len(following_screen_names)} *****")
        return following_screen_names

    def preprocess(self, sent):
        """개행 문자, URL, @id, 앞뒤 공백 제거
        Args:
            sent (str): 전처리할 문장
        Returns:
            sent (str): 전처리된 문장
        """
        sent = sent.replace("\n", " ")
        sent = sent.replace("  ", " ")
        # sent = re.sub(r"http\S+", "", sent) # URL은 활용할 지도 모르므로 우선 제거하지 않도록 주석처리
        # sent = re.sub(r"https\S+", "", sent)
        sent = re.sub(r"@\S+", "", sent)
        sent = sent.lstrip()
        sent = sent.rstrip()
        return sent

    def check_filepath(self, save_path):
        """데이터 저장 경로 폴더가 존재하는지 확인하고 없다면 생성
        Args:
            save_path (str): 저장 경로
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def __call__(self, screen_name):
        self.check_filepath(self.save_path)

        self.screen_name = screen_name
        self.screen_names = self.get_following_screen_names(screen_name)

        result = {}

        print("***** 크롤링을 시작합니다 *****")
        for screen_name in tqdm(self.screen_names):
            try:
                tweets = self.api.user_timeline(
                    screen_name=screen_name,
                    exclude_replies=False,
                    include_rts=False,
                    count=300,
                )

                for tweet in tweets:
                    replies = self.api.search_tweets(
                        q=f"to:{tweet.user.screen_name}",
                        since_id=tweet.id,
                        result_type="mixed",
                        count=150,
                        lang="ko",
                    )
                    preprocessed_tweet = self.preprocess(tweet.text)
                    matched_replies = []
                    for reply in replies:
                        if reply.in_reply_to_status_id == tweet.id:
                            try:
                                preprocessed_reply = self.preprocess(reply.text)
                                matched_replies.append(preprocessed_reply)

                            except Exception:
                                print("reply가 없습니다.")
                                pass

                    if matched_replies:
                        temp = {
                            "question": preprocessed_tweet,
                            "answer": matched_replies,
                        }
                        result[tweet.id] = temp

                file_name = f"tweets_{self.screen_name}.pickle"
                pickle_files = ""
                if glob.glob(f"{self.save_path}/*.pickle"):
                    pickle_files = glob.glob(f"{self.save_path}/*.pickle")

                if pickle_files and f"{self.save_path}/{file_name}" in pickle_files:
                    print(f"***** update {file_name} ******")
                    with open(f"{self.save_path}/{file_name}", "rb") as f:
                        data = pickle.load(f)
                        data.update(result)
                    with open(f"{self.save_path}/{file_name}", "wb") as f:
                        pickle.dump(data, f)
                else:
                    print(f"***** make {file_name} *****")
                    with open(f"{self.save_path}/{file_name}", "wb") as f:
                        pickle.dump(result, f)

            except Exception:
                pass
