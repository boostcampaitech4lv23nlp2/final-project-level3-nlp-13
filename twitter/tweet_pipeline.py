import os
from dataclasses import dataclass, field
from collections import namedtuple
from classes import UserTweet
from pathlib import Path

import tweepy
import typing
from dotenv import load_dotenv

load_dotenv()
TWITTER_CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET_KEY = os.environ.get("TWITTER_CONSUMER_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET_TOKEN = os.environ.get("TWITTER_ACCESS_SECRET_TOKEN")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET_KEY)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET_TOKEN)
User = namedtuple("User", "user_name user_screen_name")

@dataclass
class TwitterPipeline:
    FILE_NAME: str
    bot_username: str  # "armybot_13"

    def __post_init__(self):
        self.client = tweepy.Client(
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET_KEY,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET_TOKEN,
            bearer_token=TWITTER_BEARER_TOKEN,
            wait_on_rate_limit=True,
            return_type=dict,
        )
        self.since_id = self.retrieve_last_since_id()
        self.bot_user_id = self.get_user_info()["data"]["id"]

    def get_mentions(self):
        new_tweets = []
        mentions = self.client.get_users_mentions(id=self.bot_user_id, since_id=self.since_id, expansions=["author_id", "referenced_tweets.id"])
        if mentions["meta"]["result_count"] == 0:
            print("🔺 No new mentions")
        else:
            users = mentions["includes"]["users"]
            users = {user["id"]: User(user_name=user["username"], user_screen_name=user["name"]) for user in users}
            for data in mentions["data"]:
                message = data["text"].replace(f"@{self.bot_username}", "").strip()
                user = users[data["author_id"]]
                
                if data["author_id"] == self.bot_user_id:
                    # 우리 chatbot이 쓴 글이
                    continue

                tweet = UserTweet(user_id=data["author_id"], tweet_id=data["id"], message=message, user_name=user.user_name, user_screen_name=user.user_screen_name)
                new_tweets.append(tweet)
            self.since_id = mentions["meta"]["newest_id"]
            self.store_new_since_id(self.since_id)
        return new_tweets

    def reply_tweet(self, tweet, reply):
        self.client.create_tweet(in_reply_to_tweet_id=tweet.tweet_id, text=reply)

    def create_tweet(self, text):
        self.client.create_tweet(text='@endlessrain_dev 입닥쳐 말포이')

    def like_tweet(self, tweet):
        self.client.like(tweet.tweet_id)

    def retrieve_last_since_id(self):
        """마지막으로 확인한 id를 반환"""
        with open(self.FILE_NAME, "r") as f:
            last_since_id = int(f.read().strip())
        return last_since_id

    def store_new_since_id(self, new_since_id):
        """id값을 업데이트"""
        with open(self.FILE_NAME, "w") as f:
            f.write(str(new_since_id))

    def get_user_info(self, username=None):
        if username is None:
            username = self.bot_username
        return self.client.get_user(username=username)
