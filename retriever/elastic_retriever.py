import json
import os

import pandas as pd
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers


def make_db_data():
    # load data from huggingface dataset
    data = load_dataset("nlpotato/chatbot_twitter_ver2")
    question = data["train"]["Q"] + data["test"]["Q"]
    answer = data["train"]["A"] + data["test"]["A"]

    db_data = [{"id": i, "question": q, "answer": a} for i, (q, a) in enumerate(zip(question, answer))]
    # save data to json file
    if not os.path.exists("./retriever/data"):
        os.makedirs("./retriever/data")

    with open("./retriever/data/elastic_data_v1.json", "w", encoding="utf-8") as f:
        json.dump(db_data, f, ensure_ascii=False, indent=4)


class ElasticRetriever:
    def __init__(self):

        # connect to elastic search
        self.es = Elasticsearch("http://localhost:9200")

        # make index
        with open("./retriever/setting.json", "r") as f:
            setting = json.load(f)

        self.index_name = "chatbot"
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(index=self.index_name, body=setting)

        # load data
        if not os.path.exists("./retriever/data/elastic_data_v1.json"):  # TODO : 나중에 yaml 파일에서 데이터 경로 받아오도록 수정
            make_db_data()
        self.db_data = pd.read_json("./retriever/data/elastic_data_v1.json")

        # insert data
        helpers.bulk(self.es, self._get_doc(self.index_name))
        # for i, data in enumerate(db_data):
        #     insert_data = {"text": data}
        #     try:
        #         self.es.index(index=self.index_name, id=i, body=insert_data)
        #     except Exception as e:
        #         print(e)

        n_records = self.es.count(index=self.index_name)["count"]
        print(f"✅ 전체 데이터 개수: {n_records}")

    def _get_doc(self, index_name):
        doc = [
            {
                "_index": index_name,
                "_id": self.db_data.iloc[i]["id"],
                "question": self.db_data.iloc[i]["question"],
                "answer": self.db_data.iloc[i]["answer"],
            }
            for i in range(len(self.db_data))
        ]
        return doc

    def search(self, query, size=3):
        res = self.es.search(index=self.index_name, body={"query": {"match": {"question": query}}}, size=size)

        scores = [hit["_score"] for hit in res["hits"]["hits"]]
        questions = [hit["_source"]["question"] for hit in res["hits"]["hits"]]
        answers = [hit["_source"]["answer"] for hit in res["hits"]["hits"]]
        return scores, questions, answers


if __name__ == "__main__":
    elastic_retriever = ElasticRetriever()

    # test
    query = "태형아 나랑 결혼하자"
    scores, questions, answers = elastic_retriever.search(query)

    print(f"✅ 검색 결과 : {query}")
    print(f"1️⃣ 유사 query: {questions[0]}")
    print(f"답변 : {answers[0]} / 점수 : {scores[0]}")
    print(f"2️⃣ 유사 query: {questions[1]}")
    print(f"답변 : {answers[1]} / 점수 : {scores[1]}")
    print(f"3️⃣ 유사 query: {questions[2]}")
    print(f"답변 : {answers[2]} / 점수 : {scores[2]}")
