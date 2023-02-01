import argparse
import json
import os
import random
import re
import warnings

import pandas as pd
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")

# 데이터 format : {"id": 0, "intent": "질문.생일", "question": "{멤버} 언제 태어났어?", "answer": "{멤버} 생일은 {생일}이야!"}
def make_db_data():
    # read csv file
    data = pd.read_csv("./chatbot/retriever/template.csv")
    intent = data["intent"]
    question = data["Q"]
    answer = data["A"]

    db_data = [{"id": i, "intent": it, "question": q, "answer": a} for i, (it, q, a) in enumerate(zip(intent, question, answer))]
    # save data to json file
    if not os.path.exists("./chatbot/retriever/data"):
        os.makedirs("./chatbot/retriever/data")

    with open(config.data.db_path, "w", encoding="utf-8") as f:
        json.dump(db_data, f, ensure_ascii=False, indent=4)


# def make_db_data():
#     # load data from huggingface dataset
#     data = load_dataset(config.data.hugging_face_path)
#     question = data["train"]["Q"] + data["test"]["Q"]
#     answer = data["train"]["A"] + data["test"]["A"]

#     db_data = [{"id": i, "question": q, "answer": a} for i, (q, a) in enumerate(zip(question, answer))]
#     # save data to json file
#     if not os.path.exists("./chatbot/retriever/data"):
#         os.makedirs("./chatbot/retriever/data")

#     with open(config.data.db_path, "w", encoding="utf-8") as f:
#         json.dump(db_data, f, ensure_ascii=False, indent=4)


class ElasticRetriever:
    def __init__(self, config):

        # connect to elastic search
        self.es = Elasticsearch("http://localhost:9200")

        # make index
        with open(config.setting.path, "r") as f:
            setting = json.load(f)

        self.index_name = "chatbot"
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(index=self.index_name, body=setting)

        # load data
        if not os.path.exists(config.data.db_path):
            make_db_data()
        self.db_data = pd.read_json(config.data.db_path)

        # insert data
        helpers.bulk(self.es, self._get_doc(self.index_name))

        n_records = self.es.count(index=self.index_name)["count"]

    def _get_doc(self, index_name):
        doc = [
            {
                "_index": index_name,
                "_id": self.db_data.iloc[i]["id"],
                "intent": self.db_data.iloc[i]["intent"],
                "question": self.db_data.iloc[i]["question"],
                "answer": self.db_data.iloc[i]["answer"],
            }
            for i in range(len(self.db_data))
        ]
        return doc

    def search(self, query, size=3):
        res = self.es.search(index=self.index_name, body={"query": {"match": {"question": query}}}, size=size)

        scores = [hit["_score"] for hit in res["hits"]["hits"]]
        intent = [hit["_source"]["intent"] for hit in res["hits"]["hits"]]
        questions = [hit["_source"]["question"] for hit in res["hits"]["hits"]]
        answers = [hit["_source"]["answer"] for hit in res["hits"]["hits"]]
        return {"scores": scores, "intent": intent, "questions": questions, "answers": answers}


def find_member(query):
    # fmt: off
    member_dict = {
        "정국": ["정국", "전정국", "정구기", "정꾸기", "구기", "톡희", "전봉장", "전졍국", "정꾸", "전증구기", "꾸꾸", "정큑", "정궁이", "졍구기"],
        "지민": ["지민", "박지민", "지미니", "뾰아리", "쨔만", "쮀멘", "줴멘", "민", "지미나", "찌미나", "박디민", "바찌미", "짜마니", "쨔마니", "디밍", "디민", "딤인", "짐니", "자마니", "찜니", "짐쨩", "딤읭이", "박짐"],
        "남준": ["RM", "랩몬", "랩몬스터", "김남준", "남준이", "주니", "남준", "남주니", "쮸니", "남듀니", "핑몬"],
        "진": ["슥찌", "진", "석찌니", "석지니", "석진", "김석진", "햄찌", "지니"],
        "슈가": ["민윤기", "슈가", "윤기", "뉸기", "미늉기", "융긔", "늉기", "슉아", "민피디", "민군"],
        "제이홉": ["정호석", "제이홉", "호석", "호비", "호서기", "호시기", "호서긱", "홉"],
        "뷔": ["김태형", "뷔", "태형", "태태", "텽이", "태깅", "태효이", "티롱이", "쀠", "티횽이"],
    }
    # fmt: off
    for db_name, member_list in member_dict.items():
        for member in member_list:
            if member in query:
                re.sub(member, "{멤버}", query)
                return {"db_name": db_name, "call_name": member, "query": query}
    return {"db_name": None, "call_name": None, "query": query}


def find_intent(query):
    intent_dict = {
        "키": ["키"],
        "나이": ["나이", "몇 살"],
        "생일": ["생일"],
        "태어난곳": ["태어난 곳", "태어난곳", "출생지", "태어났"],
        "고향": ["고향"],
        "소속사": ["소속사", "소속"],
        "국적": ["국적", "어느나라 사람"],
        "본관": ["본관"],
        "본명": ["본명"],
        "혈액형": ["혈액형"],
        "출신(초등학교)": ["초등학교 출신", "초등학교"],
        "출신(중학교)": ["중학교 출신", "중학교"],
        "출신(고등학교)": ["고등학교 출신", "고등학교"],
        "출신(대학교)": ["대학교 출신", "대학교"],
        "전공": ["전공", "무슨 과", "무슨과", "학과"],
        "별명": ["별명", "애칭"],
        "소속사": ["소속사", "소속"],
        "데뷔년도": ["데뷔", "데뷔년도", "데뷔일", "데뷔날"],
        "SNS 링크": ["인스타 주소", "인스타 링크", "SNS 링크", "SNS 주소"],
        "SNS 아이디": ["인스타 아이디", "SNS 아이디"],
        "팔로워 수": ["팔로워 수", "팔로워수", "팔로워"],
        "취미": ["취미"],
        "발사이즈": ["발사이즈", "발 사이즈"],
        "반려동물": ["반려동물", "애완동물"],
        "가족관계": ["가족관계"],
        "예명": ["예명"],
        "종교": ["종교"],
        "훈장": ["훈장"],
        "포지션": ["포지션", "역할", "그룹 내 담당", "담당"],
        "영어이름": ["영어이름", "영어 이름", "본명 영문", "본명 영어로", "본명을 영어로"],
        "첫 앨범": ["첫 앨범", "첫 앨범 이름", "첫 앨범 제목", "첫 앨범 노래", "첫 앨범 노래 제목", "첫 앨범 노래 이름"],
    }
    for intent, keywords in intent_dict.items():
        for keyword in keywords:
            if keyword in query:
                return {"intent": intent}
    return {"intent": None}


def choose_answer_template(db_outputs, query_intent):
    # query intent와 db_outputs의 일치하는 intent가 있는지 확인 & score 10점 이상
    for i in range(len(db_outputs)):
        if db_outputs["intent"][i].split(".")[1] == query_intent and db_outputs["scores"][i] >= 10:
            answer_candidates = db_outputs["answers"][i].split(",")
            # choose answer randomly
            final_answer = random.choice(answer_candidates)
            return final_answer
    return None


def fill_answer_slot(answer_template, db_name, call_name):
    answer_template = answer_template.replace("{멤버}", call_name)

    # '{'로 시작하고 '}'로 끝나는 slot 찾기
    slots = re.findall(r"\{.*?\}", answer_template)

    # slot에 해당하는 정보 찾기 => db.json에서 가져오기
    for slot in slots:
        db_json = json.load(open("./chatbot/retriever/data/db.json", "r", encoding="utf-8"))
        slot_info = db_json[db_name][slot[1:-1]]
        answer_template = answer_template.replace(slot, slot_info)

    return answer_template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="retriever_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./chatbot/retriever/{args.config}.yaml")
    elastic_retriever = ElasticRetriever(config)

    # test
    query = input("query를 입력해주세요: ")

    # member slot이 있는지 찾기
    outputs = find_member(query)
    query = outputs["query"]  # 치환된 query
    call_name = outputs["call_name"]  # query 상의 호칭된 멤버 이름
    db_name = outputs["db_name"]  # db 검색용 멤버 이름

    # intent 찾기
    outputs = find_intent(query)
    query_intent = outputs["intent"]  # query 상의 intent

    # Elastic Search
    db_outputs = elastic_retriever.search(query)

    # query에 intent가 있는 경우
    if query_intent != None:
        # 1.1 answer template 선정 (intent가 일치해야하며 score가 10점 이상)
        final_answer = choose_answer_template(db_outputs, query_intent)
        if final_answer != None:
            # 2. answer template의 {slot}에 db로부터 찾은 정보를 채워넣기
            filled_final_answer = fill_answer_slot(final_answer, db_name, call_name)
            print(filled_final_answer)
        # 1.2 적합한 answer template이 없는 경우 생성모델에게
        else:
            print("생성 모델로 보냄")
    else:
        print("생성 모델로 보냄")
