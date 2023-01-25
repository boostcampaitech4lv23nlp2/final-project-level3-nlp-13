import pickle
import os
import re
import time
import datetime
from pathlib import Path
from typing import List, Union, Optional

import requests as req
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NaverCrawler:
    def __init__(self, runtime: str = ""):
        """
        환경에 맞는 Chrome driver가 프로젝트 최상위 폴더에 있어야 함.
        """
        self.save_path = "data/raw_data/naver"
        self.runtime = runtime
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"
        }
        self.tagger = None

    def get_news_urls(
        self, query: str = "bts", start: int = 1, since: str = "", until: str = ""
    ) -> List:
        """
        네이버 검색을 사용하여 query 기사들 link 추출. 사실상 중복되는 기사들을 수집하는 것을 최소화하기 위해
        관련뉴스 중 최상위 뉴스 하나만 추출.
        [Args]
            - query: query string
            - start: the sub-page index increasing by 10 from 1
            - since, until: time range
        """
        start = (start - 1) * 10 + 1
        naver_search_url = f"https://search.naver.com/search.naver?where=news&sort=0&photo=0&query={query}&ds={since}&de={until}&start={start}"
        res = req.get(naver_search_url, headers=self.headers)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            naver_news_tags = soup.find_all("a", {"class": "info"})
            urls = [tag["href"] for tag in naver_news_tags]
            return urls

        else:
            return None

    def read_article(self, soup) -> Union[dict, None]:
        try:
            parsed = self.parse(soup)
            return parsed  # TO-DO: class or namedtupel for news results
        except:
            return None

    def __call__(self, query: str, n: int, since: str = "", until: str = "") -> None:
        """
        [Args]
            - query: 검색어
            - n: 크롤링할 (최대) 기사수
            - since: YYYY-MM-DD. 검색기간 시작일
            - until: YYYY-MM-DD. 검색기간 마지막일
        """
        output = {
            "info": {
                "query": query,
                "runtime": self.runtime,
            },
            "data": [],
        }

        pbar = tqdm(total=n, desc="Reading newspapaer")
        start = 1
        stack = 0
        stale_soups = []
        while stack < n:
            urls = self.get_news_urls(query, start, since, until)

            if urls is None or len(urls) == 0:
                # if there is no article or reqeust fails, stop crawling
                print("Early Stopping")
                break

            for url in urls:
                time.sleep(2.1)
                if "news.naver.com" not in url:
                    continue
                res = req.get(url, headers=self.headers)
                soup = BeautifulSoup(res.text, "html.parser")
                if url in stale_soups:
                    # 직전에 추출한 기사 재추출 방지
                    continue
                parsed = self.read_article(soup)
                if parsed:
                    item = {"id": f"naver_{query}_{stack}"}
                    item.update(parsed)
                    output["data"].append(item)
                    pbar.update(1)
                    stack += 1
                    stale_soups = urls[:]

            start += 1

        pbar.close()
        print(
            f"Crawled {len(output['data'])} articles from the given query '{query}'"
        )  # TO-DO: change to logger
        self.save_pickle(query=query, run_time=self.runtime, data=output)

    def parse(self, soup) -> dict:
        """
        기사(뉴스 또는 연예뉴스)에 따라 다른 tag을 갖고 있기 때문에 달리 parse
        """
        try:
            # entertain news
            title = soup.select_one("h2", {"class": "ent_tit"}).text.strip()
            body = soup.select_one("div.article_body").text.strip()
            written_at = soup.select_one("span > em").text.strip()
            writer = soup.select_one("p.byline_p > span").text.strip()
            caps = soup.find_all("em", {"class": "img_desc"})
            if caps:
                img_captions = [cap.text.strip() for cap in caps]

        except:
            # news
            title = soup.select_one("h2.media_end_head_headline").text.strip()
            body = soup.select_one("div._article_content").text.strip()
            written_at = soup.select_one("span._ARTICLE_DATE_TIME").text.strip()
            writer = soup.select_one("em.media_end_head_journalist_name").text.strip()
            caps = soup.find_all("em", {"class": "img_desc"})
            if caps:
                img_captions = [cap.text.strip() for cap in caps]

        parsed = {
            "title": title,
            "body": body,
            "written_at": written_at,
            "writer": writer,
            "caption": img_captions,
        }
        return parsed

    def preprocess(self, raw_data_path: Optional[str] = None):
        """
        1차로 기사 제목을 기준으로 중복 제거. 동일한 columns을 가진 데이터들만 처리 가능
        csv를 인풋 path에 저장
        """

        raw_data_path = self.pickle_path if raw_data_path is None else raw_data_path
        path = Path(raw_data_path)
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = path.glob("**/*.pickle")

        ls = []
        for p in paths:
            with p.open("rb") as f:
                saved = pickle.load(f)
                df = pd.DataFrame(saved["data"])
                df.update(saved["data"])
                df["from"] = str(p)
                ls.append(df)

        df = pd.concat(ls)
        df.sort_values(by="written_at", ascending=False, inplace=True)
        start = time.time()
        df = df.apply(lambda row: pd.Series(self.preprocess_example(row)), axis=1)
        df = self.drop_duplicates(
            df, max_features=1024, alpha=1.5, beta=0.8, threshold=0.6
        )

        print(f"Took {time.time()-start}s for preprocessing")
        self.save_csv(df, "naver_corpus_1st")

    def preprocess_example(self, example: dict) -> dict:
        title, body, img_captions, writer, written_at, _from = (
            example["title"],
            example["body"],
            example["caption"],
            example["writer"],
            example["written_at"],
            example["from"],
        )
        title, body = title.strip(), body.strip()
        written_at = written_at.split()[0]

        output = {
            "title": title,
            "body_unprocessed": body,
            "body": None,
            "written_at": written_at,
            "from": _from.split("/")[-1].replace(".pickle", ""),
        }

        if self.is_photo_article(title, body) or not self.is_kor_article(title):
            return output
        body = self.remove_caption(body, img_captions)
        body = self.fix_encoded(body)
        body = self.remove_garbage(body)
        body = self.remove_info(body)
        if body.strip() == "":
            return output
        output["body"] = body
        return output

    def is_kor_article(self, title: str) -> bool:
        if re.search(r"[가-힣]", title):
            return True
        return False

    def is_photo_article(self, title: str, body: str) -> bool:
        if re.search(r"(포토\s?(?!카드)|영상|사진)", title) and len(body.split("\n")) <= 2:
            return True
        return False

    def remove_info(self, body: str) -> str:
        """
        \\n으로 split 했을 때 구두점으로 끝나지 않으면 기사의 일부분이 아니라고 간주하고 삭제
        """
        parts = body.split("\n")
        puncs = (".", "!", "?")
        cleaned = []
        for idx, part in enumerate(parts):
            part = self.remove_info_head(part.strip(" "))
            if part.strip() == "":
                continue
            if not part.endswith("다."):
                part = self.remove_info_tail(part)
            part = re.sub(r"(?<=[가-힣])다\.(?=\w)", "다. ", part)
            part = re.sub(r"^다\.\s*$", "", part)
            cleaned.append(part)

        return "\n".join(cleaned)

    # def remove_info(self, body: str) -> str:
    #     """
    #     \\n으로 split 했을 때 구두점으로 끝나지 않으면 기사의 일부분이 아니라고 간주하고 삭제
    #     """
    #     parts = body.split("\n")
    #     print(parts)
    #     puncs = (".", "!", "?")
    #     cleaned = []
    #     for idx, part in enumerate(parts):
    #         if part == "":
    #             continue
    #         part = part.strip(" ")
    #         if idx != len(parts) - 1 and not part.endswith(puncs):
    #             # e.g. ""기사내용 요약 방탄 음원 1위""
    #             continue
    #         part = self.remove_info_head(part)
    #         if idx == len(parts) -1:
    #             part = self.remove_info_tail(part)
    #             # if not part.endswith(puncs):
    #             #     continue
    #         cleaned.append(part)

    #     return "\n".join(cleaned)

    def remove_info_head(self, line: str) -> str:
        """
        서두에 있는 언론사명, 기자명 등을 제거

        """
        p = re.compile(
            r"^(\([가-힣a-zA-Z0-9=\- ]+\)|\[[가-힣a-zA-Z0-9=\- ]+\]|[가-힣a-zA-Z0-9\- ]+=)+"
        )
        line = re.sub(p, "", line)

        # p1 = re.compile("[^\.]+[\[\(].+[\]\)].+기자 =")  # [서울=신문사] 똉땡이 기자 =
        # p2 = re.compile("\[[^\.]+기자\]")  # [신문사=땡땡이 기자]
        # p3 = re.compile("[\[\(].+[\]\)] =")  # (서울=뉴스) =
        # p4 = re.compile("^\[.+?\]")  # [신문사]
        # for p in [p1, p2, p3, p4]:
        #    line = re.sub(p, "", line)
        # m = re.match(p4, line)
        # if m:
        #    line = line[m.span()[1] :]
        return line

    def remove_info_tail(self, line: str) -> str:
        """
        후미의 기자명, 언론사명을 제거
        email: ([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,})
        url: https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//  =]*)

        """
        line = (
            "다. ".join(re.findall(r"(?<=다\.).+?(?=다\.)", "다. " + line.strip())) + "다."
        )

        # p = re.compile(
        #     r"(?<=다\.)\s?[^\.]*(\(?[가-힣a-zA-Z ]+\)?)\s?(\(?\/?([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,})\)?)?\s?(\([가-힣 ]+\))?$"
        # )
        # line = re.sub(p, "", line)
        # line = re.sub(r"(?<=다\.)\s?[▲△▶][^\.]+$", "", line)  # 수상 내역 등 정보 나열
        return line

    def remove_garbage(self, body: str) -> str:
        """
        언론사별 기사와 상관없는 홍보성 문구 제거

        """
        garbage = [
            "※CBS노컷뉴스는 여러분의 제보로 함께 세상을 바꿉니다. 각종 비리와 부당대우, 사건사고와 미담 등 모든 얘깃거리를 알려주세요.이메일 : jebo@cbs.co.kr카카오톡 : @노컷뉴스사이트 : https://url.kr/b71afn",
            "<뉴미디어팀 디그(dig)>[뉴스 쉽게보기]는 매일경제 뉴미디어팀 '디그(dig)'의 주말 연재물입니다. 디그가 만든 무료 뉴스레터를 구독하시면 술술 읽히는 다른 이야기들을 월·수·금 아침 이메일로 받아보실 수 있습니다. '매일경제 뉴스레터'를 검색하고, 정성껏 쓴 디그의 편지들을 만나보세요. 아래 주소로 접속하셔도 구독 페이지로 연결됩니다.https://www.mk.co.kr/newsletter/",
            "* YTN star에서는 연예인 및 연예계 종사자들과 관련된 제보를 받습니다. ytnstar@ytn.co.kr로 언제든 연락주시기 바랍니다. 감사합니다.",
            "발로 뛰는 더팩트는 24시간 여러분의 제보를 기다립니다.▶카카오톡: '더팩트제보' 검색▶이메일: jebo@tf.co.kr▶뉴스 홈페이지: http://talk.tf.co.kr/bbs/report/write",
            "[사진]OSEN DB.",
            "[사진]OSEN DB",
        ]
        for garb in garbage:
            body = re.sub(re.escape(garb), "", body)
        return body

    def remove_caption(self, text: str, captions: List) -> str:
        """
        기사 본문 text에 포함된 이미지 caption을 제거함.
        """
        for caption in captions:
            text = re.subn(re.escape(caption), "", text, 1)[0]
        # captions = [re.escape(cap) for cap in captions]
        # p = re.compile("(" + "|".join(captions) + ")")
        # text = re.sub(p, "", text)
        return text

    def fix_encoded(self, text) -> str:
        return re.sub("\xa0", "", text)

    def drop_duplicates(
        self,
        df: pd.DataFrame,
        max_features: int = 1024,
        alpha: float = 1.5,
        beta: float = 0.8,
        threshold: float = 0.6,
    ) -> pd.DataFrame:
        """
        제목의 tfidf vector로 계산한 cosine similarity에 날짜로 만든 가중치(기사 작성일이 하루 차이면 1.5, 이외 경우는 0.8)를 곱해 기준치(0.6)을 넘으면 중복 테마의 기사로 간주하고 첫 기사만 남겨두고 제거.

        """
        self.tagger = self.get_tagger()

        # drop same articles w/ exactly same titles
        df.drop_duplicates(subset=["title", "body"], inplace=True)
        df.dropna(axis=0, how="any", inplace=True)

        # drop articles w/ same topics
        df["title"] = df["title"].apply(lambda x: self.normalize(x))
        texts = df["title"].to_numpy()
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            tokenizer=self.tokenize,
        )
        dtm = vectorizer.fit_transform(texts)
        dates = [
            datetime.date(*map(int, date[:-1].split("."))) for date in df.written_at
        ]
        cosine_simils = cosine_similarity(dtm, dtm)

        outs = []
        dup_indices = []
        for idx, simils_per_ex in enumerate(cosine_simils):
            if idx in dup_indices:
                continue
            indices_ranked, scores = self.rank(idx, cosine_simils, dates, alpha, beta)

            dup_ls = [
                ind for ind in indices_ranked if scores[ind] >= threshold and ind != idx
            ]
            dup_indices.extend(dup_ls)
            outs.append(
                {
                    "title": df["title"].iloc[idx],
                    "body": df["body"].iloc[idx],
                    "written_at": df["written_at"].iloc[idx],
                    "from": df["from"].iloc[idx],
                }
            )
        new_df = pd.DataFrame.from_dict(outs)
        return new_df

    def normalize(self, text: str) -> str:
        """
        기사 제목을 tf-idf 벡터로 만들기 전에 정규화.

        """
        text = re.sub("BTS", "방탄소년단", text)
        text = re.sub(r"(^\[.+?\]|\[.+?\]$)", "", text)
        text = re.sub(r"[^가-힣0-9A-Za-z一-龥 ]", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"(?<=[가-힣])X(?=[가-힣])", " ", text)
        return text

    def get_tagger(self):
        from kiwipiepy import Kiwi

        kiwi = Kiwi()
        for word in ["방탄소년단", "진", "정국", "지민", "RM", "슈가", "제이홉", "뷔"]:
            kiwi.add_user_word(word, "NNP")
        return kiwi

    def tokenize(self, text):
        assert self.tagger is not None, "Tagger must be initialized first"
        return [token.form for token in self.tagger.tokenize(text)]

    def rank(
        self,
        idx: int,
        cosine_simils: np.ndarray,
        dates: List[datetime.date],
        alpha: float,
        beta: float,
    ):
        """
        기준 date(criterion)과 dates를 비교. 차이가 작을수록 점수를 높여줌.
        """
        criterion = dates[idx]
        diffs = [abs(criterion - d) for d in dates]
        diffs = np.array(
            [alpha if diff <= datetime.timedelta(days=1) else beta for diff in diffs]
        )
        scores = cosine_simils[idx] * diffs  # element-wise multiplication
        indices = np.argsort(-scores)

        return indices, scores

    def save_pickle(self, query: str, run_time: str, data: dict) -> None:
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.pickle_path = os.path.join(
            self.save_path, f"{query}_{run_time}_size{len(data['data'])}.pickle"
        )
        with open(self.pickle_path, "wb") as f:
            if len(data["data"]) > 0:
                pickle.dump(data, f)
                print(f"Saved to {self.pickle_path}")

    def save_csv(self, df: pd.DataFrame, save_name: str) -> None:
        save_path = "data/preprocessed_data/naver/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        csv_path = os.path.join(save_path, f"{save_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved_to {csv_path}")


# from dotenv import load_dotenv

# load_dotenv()
# NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
# NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")
#
# class NaverCrawler:
# """
# Naver Search API로 (최대 100개) 기사의 url을 받아온 후 셀레니움으로 개별 기사를 태깅
# 검색 조건(언론사, 기사유형, 기간)등을 지정할 수 없고, 중복되는 기사들이 너무 많아 나중에 daily 업데이트용으로 바꿀 예정
# """
#     def __init__(self, runtime: str):
#         self._headers = {
#             "X-Naver-Client-Id": NAVER_CLIENT_ID,
#             "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
#             "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
#         }

#         self.save_path = "data/raw_data/naver"
#         self.runtime = runtime
#         self.driver = webdriver.Chrome("./chromedriver")

#     def get_news_urls(self, query="bts", start=1, display=100) -> List:
#         """
#         네이버 검색 API를 사용하여 query 기사들 link 추출
#         """
#         naver_search_url = f"https://openapi.naver.com/v1/search/news.json?query={query}&start={start}&sort=sim&display={display}"
#         res = req.get(naver_search_url, headers=self._headers)
#         if res.status_code == 200:
#             res = res.json()
#             items = res["items"]
#             urls = [item["link"] for item in items if "news.naver.com" in item["link"] or "entertain.naver.com" in item["link"]]
#             print(f"Got {len(urls)} urls from Naver.")

#         else:
#             raise Exception(f"Failed to get response from Naver API. Code: {res.status_code}")  # TO-DO: use logger

#         return urls

#     def read_news(self, url: str) -> Union[dict, None]:
#         try:
#             self.driver.get(url)
#             self.driver.implicitly_wait(10)
#             parsed = self.parse()
#             # time.sleep(3.2)
#             return parsed # TO-DO: class or namedtupel for news results
#         except:
#             return None

#     def __call__(self, query, n) -> None:
#         urls = self.get_news_urls(query=query, display=n)
#         output = {
#             "info": {
#                 "query": query,
#                 "runtime": self.runtime,
#             },
#             "data": [],
#         }
#         for idx in trange(len(urls)):
#             url = urls[idx]
#             news = self.read_news(url)
#             if isinstance(news, dict):
#                 item = {"id": f"naver_{query}_{idx}", "url": url}
#                 item.update(news)
#                 output["data"].append(item)

#         self.driver.quit()
#         print(f"Crawled {len(output['data'])} articles from the given query '{query}'")  # TO-DO: change to logger
#         self.save(query=query, run_time=self.runtime, data=output)

#     def parse(self) -> dict:
#         """
#         기사(뉴스 또는 연예뉴스)에 따라 다른 tag을 갖고 있기 때문에 달리 parse
#         """
#         by = By.CSS_SELECTOR
#         try:
#             # entertain news
#             title = self.driver.find_element(by, "h2.end_tit").text.strip()
#             body = self.driver.find_element(by, "div.article_body").text.strip()  # class: article_body
#             if caps:= self.driver.find_elements(by, "em.img_desc"):
#                 img_captions = [cap.text.strip() for cap in caps]
#             written_at = self.driver.find_element(by, "span > em").text.strip()
#             writer = self.driver.find_element(by, "p.byline_p > span").text.strip()

#         except:
#             # news
#             title = self.driver.find_element(by, "h2.media_end_head_headline").text.strip()
#             body = self.driver.find_element(by, "div._article_content").text.strip()
#             if caps:= self.driver.find_elements(by, "em.img_desc"):
#                 img_captions = [cap.text.strip() for cap in caps]
#             written_at = self.driver.find_element(by, "span._ARTICLE_DATE_TIME").text.strip()
#             writer = self.driver.find_element(by, "em.media_end_head_journalist_name").text.strip()

#         parsed = {
#             "title": title,
#             "body": body,
#             "written_at": written_at,
#             "writer": writer,
#             "caption": img_captions,
#         }
#         return parsed

#     def preprocess(self, title, body, img_captions, written_at, writer):
#         body = self.remove_caption(body, img_captions)

#         return title, body, written_at, writer

#     def remove_caption(self, text:str, captions:List) -> str:
#         """
#         기사 본문 text에 포함된 이미지 caption을 제거함.
#         """
#         for caption in captions:
#             pattern = re.compile(caption)
#             text = re.sub(pattern, "", text)
#         return text
