import re
import json
import typing

from dataclasses import dataclass, field
from pathlib import Path

from kiwipiepy import Kiwi, Sentence, Token


@dataclass
class DataPipeline:
    log_dir: str
    special_tokens: field(default_factory=list)

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)
        self.tagger = Kiwi(
            model_type="sbg",
            typos="basic",
        )
        for token in self.special_tokens:
            self.tagger.add_user_word(token, "NNP")

    def postprocess(self, text:str, user_screen_name) -> str:
        if "<account>" in text:
            text = re.sub("<account>", user_screen_name, text) 
        return text

    def preprocess(self, query: str) -> str:
        query = self.get_clean_text(query)
        query = self.normalize(query)
        return query

    def get_clean_text(self, query: str, n: int = 1) -> str:
        cleaned = []
        query = self.remove_invalid_chrs(query)
        query = self.remove_duplicates(query, n=n)
        cleaned.append(query)
        return cleaned

    def remove_invalid_chrs(self, query: str) -> str:
        """한글, 알파벳, 숫자, 물음표, 스페이스 외 제거"""
        return re.sub(r"[^ㄱ-힣A-Za-z0-9\? ]", "", query).strip()

    def remove_duplicates(self, query: str, n: int) -> str:
        """단독 자모음 또는 구두점 중복을 제거하여 연속되는 n개로 만듦"""
        return re.sub(r"([ㄱ-ㅣ\?])\1+", r"\1" * n, query)

    def normalize(self, query: str) -> typing.List[Sentence]:
        """
        1. 띄어쓰기 교정, 간단한 오탈자 교정 후
        2. 문장 분리 하여 리턴
        """
        normalized = []
        query = self.correct_spacing(query)
        sents = self.split_into_sentences(query)
        ls = []
        for sent in sents:
            sent = self.tokenize(sent)
            ls.append(sent)
            normalized.append(ls)
        return normalized

    def correct_spacing(self, sent: str) -> str:
        return self.tagger.space(sent)

    def split_into_sentences(self, sent: str) -> typing.List[Sentence]:
        return self.tagger.split_into_sents(
            sent, normalize_coda=False, return_tokens=True
        )

    def tokenize(self, sentence: Sentence) -> str:
        sent = " ".join([token.form for token in sentence.tokens])
        return sent

    def correct_grammar(self, retriever_output) -> str:
        variants = {
            "가": "이",
            "를": "을",
            "는": "은",
            "야": "이야",
            "지": "이지",
            "랑": "이랑",
            "예요": "이에요",
            "는요": "은요",
            "로": "으로",
            "로는": "으로는",
            "로요": "으로요",
            "로는요": "으로는요",
        }
        names_with_coda = ["박지민", "지민", "김남준", "남준", "진", "석진", "김석진", "제이홉", "정호석", "호석", "태형", "김태형", "정국", "전정국"]
        sent = retriever_output.query
        target = retriever_output.db_name
        if target not in names_with_coda:
            return sent

        for match in re.findall(f"{target}(?=(가|를|는|야|예요|는요|로|로는|로요|로는요))", sent):
            sent = re.sub(target + match, target + variants[match], sent)

        return sent

    def _analyze_chr(self, character):
        cc = ord(character) - 44032
        onset = cc // (21 * 28)
        coda = cc % 28
        return {"onset": onset, "coda": coda}

    def log(self, new_entries: typing.List[dataclass], save_name: str):
        """
        Log new entries in dataclass formats
        """
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True)

        log_path = self.log_dir / f"{save_name}.json"
        with log_path.open("a+", encoding="utf-8") as f:
            try:
                file = json.load(f)
            except:
                file = dict()
            for entry in new_entries:
                file.update(entry.__dict__)
            json.dump(file, f, indent=4, ensure_ascii=False)
