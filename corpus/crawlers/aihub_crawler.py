import json
import typing
import re
import pandas as pd
from pathlib import Path
from tqdm import trange
from kiwipiepy import Kiwi


class NewsCrawler:
    """
    AIHub 대규모 웹데이터 기반 한국어 말뭉치 데이터: <연예> 뉴스

    """

    def __init__(self, path):
        self.path = Path(path)
        self.tagger = None

    def get_file_paths(self) -> typing.List:
        return self.path.glob("**/*.json")

    def __call__(self) -> None:
        paths = self.get_file_paths()
        ls = []
        for path in paths:
            items = self.read_json(path)
            ls.append(pd.DataFrame(items))
        df = pd.concat(ls, ignore_index=True)
        self.save(df, "aihub_news")

    def read_json(self, path: str) -> dict:
        with path.open("rb") as f:
            file = json.load(f)
            source_id = "aihub_news_" + file["header"]["source_file"]
            ids, titles, bodies = [], [], []
            for idx, doc in enumerate(file["named_entity"], start=1):
                _id = source_id + f"_{idx}"
                title = doc["title"][0]["sentence"]
                body = self.read_body(doc["content"])
                p = re.compile(r"(?<=다)\.\.")
                body = re.sub(p, ".", body)
                ids.append(_id)
                titles.append(title)
                bodies.append(body)

        return {
            "id": ids,
            "title": titles,
            "body": bodies,
        }

    def read_body(self, body: typing.List) -> str:
        sentences = [sent_dict["sentence"] for sent_dict in body]
        return " ".join(sentences)

    def save(self, df: pd.DataFrame, save_name):
        save_path = self.path / f"{save_name}.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved to {str(save_path)}")

    def filter_bts(self, df: pd.DataFrame):
        """
        Get articles with titles containing "BTS" or "bts" or "방탄소년단"
        """
        return df[df["title"].str.contains("(BTS|bts|방탄소년단)", regex=True, na=False)]

    def preprocess(self, df: pd.DataFrame):
        self.tagger = Kiwi()
        df["title"] = df["title"].apply(self.preprocess_title)
        df["body"] = df["body"].apply(self.preprocess_body)
        df.dropna(axis=0, how="any", subset="body", inplace=True)
        df["text"] = df.apply(lambda row: row["title"] + " " + row["body"], axis=1)
        # df = df[~df["text"].str.contains("(이름)", regex=False)]
        df.to_csv("aihub_news_bts_preprocessed.csv")

    def preprocess_title(self, text):
        return re.sub(r"[\(\[].+?[\)\]]", "", text)

    def preprocess_body(self, text):
        sents = [
            sent.text.replace(";", " ")
            for sent in self.tagger.split_into_sents(text)
            if sent.text.strip().endswith(".")
        ]
        if len(sents) <= 2:
            return None
        text = " ".join(sents)
        if "(이름)" in text:
            return None
        text = re.sub(";", "", text)
        text = re.sub(r"^(.+?)?\s?=", "", text)
        return text


class CommentCrawler(NewsCrawler):
    """
    AI Hub 온라인 구어체 말뭉치 (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=625)
    """

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self) -> None:
        paths = self.get_file_paths()
        ls = []
        for path in paths:
            items = self.read_json(path)
            if items is not None:
                df = pd.DataFrame(items)
                ls.append(df)
        df = pd.concat(ls, ignore_index=True)
        self.save(df, "aihub_comment")

    def read_json(self, path: str) -> dict:
        """
        주제가 KPOP(코드: KP)인 대화만 수집
        """
        with path.open("rb") as f:
            try:
                file = json.load(f)
                source_id = "aihub_comment_" + file["header"]["source_file"]
                subject = file["header"]["subject"]
            except:
                # when json decoding error occurs
                print("json decoding error")
                return None

            if subject != "KP":
                return None

            ids, comments = [], []
            for idx, comment in enumerate(file["named_entity"], start=1):
                _id = source_id + f"_{idx}"
                comment = comment["content"]["sentence"]
                ids.append(_id)
                comments.append(comment)
        return {
            "id": ids,
            "comment": comments,
        }
