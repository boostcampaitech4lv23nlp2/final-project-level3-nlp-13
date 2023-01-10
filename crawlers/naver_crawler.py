import time

import pandas as pd
import requests as req
from bs4 import BeautifulSoup

# import pytz
# import datetime


class NaverCrawler:
    def __init__(self, headers):
        self._headers = headers

    def get_news_urls(self, query="bts", start=1, display=100):
        """
        네이버 검색 API를 사용하여 query 기사들 link 추출
        """
        naver_search_url = f"https://openapi.naver.com/v1/search/news.json?query={query}&start={start}&sort=sim&display={display}"
        res = req.get(naver_search_url, headers=self._headers)
        if res.status_code == 200:
            res = res.json()
            items = res["items"]
            urls = [item["link"] for item in items if "news.naver.com" in item["link"] or "entertain.naver.com" in item["link"]]

        else:
            raise Exception(f"Response failed. Code: {res.status_code}")  # TO-DO: use logger

        return urls

    def read_news(self, url):
        res = req.get(url)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            title = soup.select_one("h2", "end_tit").text.strip()
            print(title)
            body = soup.select_one("div", id="article_body").text.strip()
            print(body)
            time.sleep(0.3)
            return {
                "title": title,
                "body": body,
            }  # TO-DO: class or namedtupel for news results
        else:
            raise Exception(f"Response failed. Code: {res.status_code}")

    def __call__(self, query, n):
        urls = self.get_news_urls(query=query, display=n)
        news = [self.read_news(url) for url in urls]
        return news
