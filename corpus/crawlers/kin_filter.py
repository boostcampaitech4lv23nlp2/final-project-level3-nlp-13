import re
import pandas as pd
import json
import typing

from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Iterator
from kiwipiepy import Kiwi


@dataclass
class KinFilter:
    df: pd.DataFrame = None
    tokens_to_add: typing.List[str] = field(default_factory=list)

    def __post_init__(self):
        self.tagger = Kiwi()
        if self.tokens_to_add == []:
            self.tokens_to_add = [
                "방탄소년단",
                "진",
                "정국",
                "지민",
                "RM",
                "rm",
                "슈가",
                "제이홉",
                "뷔",
                "BTS",
                "bts",
                "빅히트",
                "아미",
                "보라해",
            ]
        for token in self.tokens_to_add:
            if token in ["보라해"]:
                POS = "IC"  # 감탄사?
            else:
                POS = "NNP"
            self.tagger.add_user_word(token, POS)

    @classmethod
    def filter_by_title(self, title: str) -> bool:
        """
        Args:
            title: 기사 제목
        Returns:
            제목으로 평가했을 때 미적합 데이터이면 False
        """
        if self.is_photo(title) or self.is_to_sell(title):
            return False
        return True

    def is_photo(self, title: str) -> bool:
        p = re.compile(r"(사진|움짤|영상|출처|셀카|원본|포토|프사|짤|GIF|gif|화질|엽사|배경|도안)")
        if re.search(p, title):
            return True
        return False

    def is_to_sell(self, title: str) -> bool:
        p = re.compile(r"(양도|하자|시세|포카|당근|택포|가격|미개봉|기스)")
        if re.search(p, title):
            return True
        return False

    def get_csv_paths(self, raw_data_path: str) -> Iterator:
        """
        Args:
            csv 파일이나 파일들이 있는 폴더 경로
        Returns:
            해당 폴더에 있는 모든 파일들의 경로
        """
        path = Path(raw_data_path)
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = path.glob("**/*.csv")
        return paths

    def preprocess(self, path: str) -> pd.DataFrame:
        paths = self.get_csv_paths(path)
        ls = []
        for path in paths:
            with path.open() as f:
                ls.append(pd.read_csv(f))

        df = pd.concat(ls)
        df.fillna("", inplace=True)
        df["text"] = df.apply(
            lambda row: self.clean(row["title"]) + " " + self.clean(row["query"]), axis=1
        )
        df["text"] = df.apply(
            lambda row: row["text"] if self.filter_by_title(row["text"]) else None,
            axis=1,
        )

        df.dropna(axis=0, how="any", subset="text", inplace=True)
        df.drop_duplicates(subset="text", inplace=True, ignore_index=True)
        df["answer"] = df["answer"].apply(self.clean)
        df = df[["text", "answer"]]
        return df

    def clean(self, text: str) -> dict:
        text = self.split_to_sents(text)
        text = self.remove_noise(text)
        return text

    def remove_noise(self, sentences: typing.List[str]) -> str:
        p = re.compile(
            r"(ㅈㄱㄴ|제곧내|내공|답변|https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)|파트너스 활동을 통해 일정액의 수수료를 제공받을 수 있음|지식인)"
        )
        nonnoise = []
        for sent in sentences:
            if re.search(p, sent):
                continue
            nonnoise.append(sent)
        return " ".join(nonnoise).strip()

    def split_to_sents(self, text: str) -> typing.List:
        return [
            sent.text
            for sent in self.tagger.split_into_sents(text, return_tokens=False)
        ]

    def save_csv(self, df: pd.DataFrame, save_name: str):
        save_path = Path("data/preprocessed_data/kin/")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        df.to_csv(save_path / save_name + ".csv", index=False)
