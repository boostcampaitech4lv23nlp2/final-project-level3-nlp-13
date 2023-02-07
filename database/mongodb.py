import pymongo
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
connection = pymongo.MongoClient(MONGO_URI)

class MongoDB:
    def __init__(self):
        self.connection = connection
        self.collection = self.connect_db("nlpotato", "twitter_bot_log")

    def connect_db(self, db_name, collection_name):
        # 만약 collection이 없다면 생성
        if collection_name not in self.connection[db_name].list_collection_names():
            self.connection[db_name].create_collection(collection_name)
        # collection 연결
        self.collection = self.connection[db_name][collection_name]
        return self.collection
    
    def insert_one(self, data):
        self.collection.insert_one(data)
