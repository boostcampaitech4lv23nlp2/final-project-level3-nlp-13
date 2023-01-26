import typing
import re
import os
import time
import requests as req
import pandas as pd

from dataclasses import dataclass
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
X_Naver_Client_Id = os.environ.get("X_Naver_Client_Id")
X_Naver_Client_Secret = os.environ.get("X_Naver_Client_Secret")

headers = {
    "X-Naver-Client-Id": X_Naver_Client_Id,
    "X-Naver-Client-Secret": X_Naver_Client_Secret,
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
}


@dataclass
class KinCrawler:
    runtime: str
    save_path: str = "data/raw_data/kin"

    def get_kin_urls(self, query: str, start: int) -> typing.List:
        start = (start - 1) * 10 + 1
        naver_search_url = (
            f"https://openapi.naver.com/v1/search/kin.json?query={query}&start={start}"
        )
        res = req.get(naver_search_url, headers=headers)
        if res.status_code == 200:
            res = res.json()
            items = res["items"]
            urls = [item["link"] for item in items]
            return urls
        else:
            None

    def __call__(self, query: str, n: int):
        pbar = tqdm(total=n, desc="Reading QnAs")
        stack = 0
        start = 1
        titles, queries, answers = [], [], []
        while stack < n:
            urls = self.get_kin_urls(query=query, start=start)

            if urls is None or len(urls) == 0:
                print("Early Stopping")
                break

            for url in urls:
                # time.sleep(1.2)
                res = req.get(url, headers=headers)
                soup = BeautifulSoup(res.text, "html.parser")

                parsed = self.read_qna(soup)
                title = parsed["title"]
                q = parsed["query"]
                for answer in parsed["answers"]:
                    titles.append(title)
                    queries.append(q)
                    answers.append(answer)
                pbar.update(1)
                stack += 1
            start += 1
        df = pd.DataFrame(
            {
                "title": titles,
                "query": queries,
                "answer": answers,
            }
        )
        self.save_csv(df, f"kin_{query}")

    def read_qna(self, soup):
        title = soup.select_one("div.title").text.strip()
        try:
            query = soup.select_one("div.c-heading__content")
        except:
            query = None
        if query is not None:
            query = re.sub(r"<.+?>", " ", str(query)).strip()

        texts = soup.select("div.se-main-container")
        answers = []
        for text in texts:
            answer = " ".join([s.text.strip() for s in text.find_all("span")])
            answers.append(answer)

        return {
            "title": title,
            "query": query,
            "answers": answers,
        }

    def save_csv(self, df: pd.DataFrame, save_name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        path = os.path.join(self.save_path, f"{save_name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
