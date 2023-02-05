import os
from dataclasses import dataclass, field
from pathlib import Path

import tweepy
import typing
from dotenv import load_dotenv

load_dotenv()
TWITTER_CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET_KEY = os.environ.get("TWITTER_CONSUMER_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET_TOKEN = os.environ.get("TWITTER_ACCESS_SECRET_TOKEN")
auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET_KEY)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET_TOKEN)


@dataclass
class TwitterPipeline:
    FILE_NAME: str
    username: str

    def __post_init__(self):
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        self.since_id = self.retrieve_last_since_id()

    def get_mentions(self):
        new_tweets = []
        for tweet in tweepy.Cursor(
            self.api.mentions_timeline, since_id=self.since_id
        ).items():
            if tweet.id <= self.since_id or self.username in tweet.text.lower():
                continue
            self.since_id = tweet.id
            new_tweets.append(tweet)

        self.store_new_since_id(self.since_id)
        return new_tweets

    def reply_tweet(self, tweet, reply):

        self.api.update_status(status=reply, in_reply_to_status_id=tweet.id)

    def retrieve_last_since_id(self):
        """마지막으로 확인한 id를 반환"""
        with open(self.FILE_NAME, "r") as f:
            last_since_id = int(f.read().strip())
        return last_since_id

    def store_new_since_id(self, new_since_id):
        """id값을 업데이트"""
        with open(self.FILE_NAME, "w") as f:
            f.write(str(new_since_id))

    def reply_to_tweets(self):
        last_seen_id = self.retrieve_last_seen_id(self.FILE_NAME)
        mentions = self.api.mentions_timeline(last_seen_id, tweet_mode="extended")
        for mention in reversed(mentions):
            last_seen_id = mention.id
            self.store_last_seen_id(last_seen_id, self.FILE_NAME)

            if self.username in mention.full_text.lower():
                input_text = mention.full_text.replace(
                    str(mention.user.screen_name), ""
                ).replace("@", "")
                print(input_text)
                return last_seen_id, str(mention.user.screen_name), input_text


@dataclass
class TwitterupdatePipeline:
    username: str
    output_text: str
    last_seen_id: str

    def __post_init__(self):
        self.api = tweepy.API(auth, wait_on_rate_limit=True)

    def update(self):
        new_status = self.api.update_status(
            "@" + self.username + " " + self.output_text, self.last_seen_id
        )
