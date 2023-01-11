import pickle
import random
import time
from typing import List, Union

import requests as req
from bs4 import BeautifulSoup
from tqdm import trange

# import pytz
# import datetime


class NaverCrawler:
    def __init__(self, headers: str, runtime: str):
        self._headers = headers
        self.save_path = "data/raw_data/naver"
        self.runtime = runtime

    def get_news_urls(self, query="bts", start=1, display=100) -> List:
        """
        네이버 검색 API를 사용하여 query 기사들 link 추출
        """
        naver_search_url = f"https://openapi.naver.com/v1/search/news.json?query={query}&start={start}&sort=sim&display={display}"
        res = req.get(naver_search_url, headers=self._headers)
        if res.status_code == 200:
            res = res.json()
            items = res["items"]
            urls = [item["link"] for item in items if "news.naver.com" in item["link"] or "entertain.naver.com" in item["link"]]
            print(f"Got {len(urls)} urls from Naver.")

        else:
            raise Exception(f"Response failed. Code: {res.status_code}")  # TO-DO: use logger

        return urls

    def read_news(self, url: str) -> dict:
        res = req.get(url)
        if res.status_code == 200:
            try:
                soup = BeautifulSoup(res.text, "html.parser")
                title = soup.select_one("h2", {"class": "end_tit"}).text.strip()
                body = soup.select_one("div.article_body").text.strip()  # class: article_body
                written_at = soup.select_one("span > em").text.strip()
                writer = soup.select_one("p.byline_p > span").text.strip()
                publisher = soup.select_one("div.press_logo").img["alt"]

                time.sleep(random.randrange(3))
                return {
                    "title": title,
                    "body": body,
                    "written_at": written_at,
                    "writer": writer,
                    "publisher": publisher,
                }  # TO-DO: class or namedtupel for news results
            except:
                print("*****************************")
                print(f"failed to parse {url}")
                return None
        else:
            raise Exception(f"Response failed. Code: {res.status_code}")

    def __call__(self, query, n) -> None:
        urls = self.get_news_urls(query=query, display=n)
        output = {
            "info": {
                "query": query,
                "runtime": self.runtime,
            },
            "data": [],
        }
        for idx in trange(len(urls)):
            news = self.read_news(urls[idx])
            if news:
                print(news)
                item = {"id": f"naver_{query}_{idx}"}.update(news)
                output["data"].append(item)

        print(f"Crawled {len(output['data'])} articles from the given query '{query}'")  # TO-DO: change to logger
        self.save(query=query, run_time=self.runtime, data=output)

    def save(self, query: str, run_time: str, data: dict) -> None:
        with open(f"{self.save_path}_{query}_{run_time}.pickle", "wb") as f:
            pickle.dump(data, f)
            print(f"Saved to {self.save_path}")
