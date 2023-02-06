import argparse
import json
import os
import random
import re
import warnings
import sys
import pandas as pd
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from classes import RetrieverOutput


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

    with open("./chatbot/retriever/data/answer_template.json", "w", encoding="utf-8") as f:
        json.dump(db_data, f, ensure_ascii=False, indent=4)


class ElasticRetriever:
    def __init__(self):

        # connect to elastic search
        self.es = Elasticsearch("http://localhost:9200")

        # make index
        with open("./chatbot/retriever/setting.json", "r") as f:
            setting = json.load(f)

        self.index_name = "chatbot"
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(index=self.index_name, body=setting)

        # load data
        if not os.path.exists("./chatbot/retriever/data/answer_template.json"):
            make_db_data()
        self.db_data = pd.read_json("./chatbot/retriever/data/answer_template.json")

        # insert data
        helpers.bulk(self.es, self._get_doc(self.index_name))
        self.es.indices.refresh(index=self.index_name)
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

    def find_member(self, query):
        # fmt: off
        member_dict = {
            "정국": ["정국", "전정국", "정구기", "정꾸기", "구기", "꾸기", "톡희", "전봉장", "전졍국", "정꾸", "전증구기", "꾸꾸", "정큑", "정궁이", "졍구기"],
            "지민": ["지민", "박지민", "지미니", "뾰아리", "쨔만", "쮀멘", "줴멘", "민", "지미나", "찌미나", "박디민", "바찌미", "짜마니", "쨔마니", "디밍", "디민", "딤인", "짐니", "자마니", "찜니", "짐쨩", "딤읭이", "박짐"],
            "RM": ["RM", "랩몬", "랩몬스터", "김남준", "남준이", "주니", "남준", "남주니", "쮸니", "남듀니", "핑몬", "알엠", "rm"],
            "진": ["슥찌", "진", "석찌니", "석지니", "석진", "김석진", "햄찌", "지니"],
            "슈가": ["민윤기", "슈가", "윤기", "뉸기", "미늉기", "융긔", "늉기", "슉아", "민피디", "민군"],
            "제이홉": ["정호석", "제이홉", "호석", "호비", "호서기", "호시기", "호서긱", "홉"],
            "뷔": ["김태형", "뷔", "태형", "태태", "텽이", "태깅", "태효이", "티롱이", "쀠", "티횽이", "V", "v"],
            "BTS": ["방탄소년단", "방탄", "BTS", "bts", "비티엣스"]
        }
        # fmt: on
        for db_name, member_list in member_dict.items():
            for member in member_list:
                if member in query:
                    query = re.sub(member, "{멤버}", query)
                    query = re.sub("데{멤버}년도", "데뷔년도", query) 
                    return {"db_name": db_name, "call_name": member, "query": query}
        return {"db_name": None, "call_name": None, "query": query}

    def find_intent(self, query):
        intent_json = json.load(open("./chatbot/retriever/data/intent_keyword.json", "r", encoding="utf-8"))

        for intent, keywords in intent_json.items():
            keywords_list = keywords["words"].split(",")
            for keyword in keywords_list:
                if keyword.strip() in query:
                    return {"intent": intent}
        return {"intent": None}

    def choose_answer_template(self, top3_outputs, query_intent):
        # query intent와 top3_outputs의 intent가 일치하면서 score 9점 이상
        for i in range(len(top3_outputs["scores"])):
            if top3_outputs["intent"][i].split(".")[1] == query_intent and top3_outputs["scores"][i] >= 9:
                answer_candidates = top3_outputs["answers"][i].split(",")
                # 랜덤하게 answer template 선택
                final_answer = random.choice(answer_candidates)
                return final_answer, top3_outputs["scores"][i]
        return None, None

    def fill_answer_slot(self, answer_template, db_name, call_name):
        # answer template에 {멤버} slot을 치환해야 하는 경우
        if call_name and "{멤버}" in answer_template:
            answer_template = answer_template.replace("{멤버}", call_name)

        # answer template에 멤버 이외의 slot 확인
        slots = re.findall(r"\{.*?\}", answer_template)

        # slot에 해당하는 정보 db.json으로부터 fill
        db_json = json.load(open("./chatbot/retriever/data/db.json", "r", encoding="utf-8"))

        for slot in slots:
            # 멤버 관련 질문인 경우
            if call_name:
                try:
                    print("💜 db_name: ", db_name)
                    if db_name == "BTS":
                        slot_info_candidate = db_json[db_name][slot[1:-1]].split(",")
                        slot_info = random.choice(slot_info_candidate).strip()
                    else:
                        slot_info = db_json[db_name][slot[1:-1]]
                    answer_template = answer_template.replace(slot, slot_info)
                except:
                    pass
            else:
                pass
                # try:
                #     slot_info_list = db_json
                #     slot_info = random.choice(slot_info_list)
                #     answer_template = answer_template.replace(slot, slot_info)
                # except:
                #     pass

        # 채우지 못한 슬롯 확인
        slots_after = re.findall(r"\{.*?\}", answer_template)
        if slots_after:
            return None
        return answer_template

    def return_answer(self, query):
        """
        Args:
            query (str): 입력 문장
        """
        # 1. 입력 query에서 member slot 추출 및 치환 : {멤버} -> 정국
        outputs = self.find_member(query)
        member_replaced_query = outputs["query"]
        call_name = outputs["call_name"] # 없으면 None
        db_name = outputs["db_name"] # 없으면 None

        # 2. 입력 query에서 intent 키워드 매칭
        outputs = self.find_intent(member_replaced_query)
        query_intent = outputs["intent"]

        # 3. 입력 query를 Elastic Search를 통해 유사 문장 top3 추출
        top3_outputs = self.search(member_replaced_query)

        # 4.1 입력 query에 intent가 있는 경우
        if query_intent:
            # 4.1.1 answer template 선정
            answer_template, bm25_score = self.choose_answer_template(top3_outputs, query_intent)
            # 4.1.2 answer template이 있는 경우
            if answer_template != None:
                # 4.1.2.1 answer_template의 slot에 db 정보 채우기
                filled_answer_template = self.fill_answer_slot(answer_template, db_name, call_name)
                return RetrieverOutput(query=filled_answer_template, bm25_score=bm25_score, db_name=db_name)
            # 4.1.3 answer template이 없는 경우 None 반환 => generation 모델에 전달
            else:
                return RetrieverOutput(query=None, bm25_score=None, db_name=None)
        # 4.2 입력 query에 intent가 없는 경우
        else:
            # Elastic Search output에서 intent가 chitchat인 경우
            if top3_outputs["intent"][0].split(".")[0] == "chitchat" and top3_outputs["scores"][0] >= 6:
                candidate_answer_templates = top3_outputs["answers"][0].split(",")
                # 랜덤하게 answer template 선택
                answer_template = random.choice(candidate_answer_templates)
                # answer_template의 slot에 db 정보 채우기
                filled_answer_template = self.fill_answer_slot(answer_template, db_name, call_name)
                return RetrieverOutput(query=filled_answer_template, bm25_score=top3_outputs["scores"][0], db_name=None)
            # => generation 모델에 전달
            else:
                return RetrieverOutput(query=None, bm25_score=None, db_name=None)



if __name__ == "__main__":
    elastic_retriever = ElasticRetriever()

    # test
    query = input("query를 입력해주세요: ")

    answer = elastic_retriever.return_answer(query)
    print(answer)
