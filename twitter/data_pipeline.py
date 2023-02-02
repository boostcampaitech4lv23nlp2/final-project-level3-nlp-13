from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import tweepy
import os
from kiwipiepy import Kiwi, Sentence, Token

load_dotenv()
TWITTER_CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET_KEY = os.environ.get("TWITTER_CONSUMER_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET_TOKEN = os.environ.get("TWITTER_ACCESS_SECRET_TOKEN")
auth  = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET_KEY)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET_TOKEN)

@dataclass
class TwitterPipeline:
    FILE_NAME: str
    username: str

    def __post_init__(self):
        self.api = tweepy.API(auth, wait_on_rate_limit=True)


    def retrieve_last_seen_id(self, file_name):
        """마지막으로 확인한 id를 반환"""
        f_read = open(file_name, 'r')
        last_seen_id = int(f_read.read().strip())
        f_read.close()
        return last_seen_id

    def store_last_seen_id(self, last_seen_id, file_name):
        """id값을 업데이트"""
        f_write = open(file_name, 'w')
        f_write.write(str(last_seen_id))
        f_write.close()
        return

    def reply_to_tweets(self):
        last_seen_id = self.retrieve_last_seen_id(self.FILE_NAME)
        mentions = self.api.mentions_timeline(last_seen_id,tweet_mode='extended')
        for mention in reversed(mentions):
            last_seen_id = mention.id
            self.store_last_seen_id(last_seen_id, self.FILE_NAME)

            if self.username in mention.full_text.lower(): 
                input_text = mention.full_text.replace(str(mention.user.screen_name), '').replace("@",'')
                return last_seen_id, str(mention.user.screen_name), input_text