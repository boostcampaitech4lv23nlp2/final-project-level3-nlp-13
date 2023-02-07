import os
from dataclasses import dataclass, field
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
            for idx in range(len(mentions["data"])):
                data = mentions["data"][idx]
                user = mentions["includes"]["users"][idx]
                message = data["text"].replace(f"@{self.bot_username}", "").strip()
                if user["id"] == self.bot_user_id:
                    # 우리 chatbot이 쓴 글이
                    continue

                tweet = UserTweet(user_id=user["id"], tweet_id=data["id"], message=message, user_name=user["username"])
                new_tweets.append(tweet)
            self.since_id = mentions["meta"]["newest_id"]
            self.store_new_since_id(self.since_id)
        return new_tweets

    def reply_tweet(self, tweet, reply):
        self.client.create_tweet(in_reply_to_tweet_id=tweet.tweet_id, text=reply)

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
