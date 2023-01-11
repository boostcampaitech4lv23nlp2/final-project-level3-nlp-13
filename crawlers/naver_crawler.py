import random
import time

import requests as req
from bs4 import BeautifulSoup

# import pytz
# import datetime


class NaverCrawler:
    def __init__(self, headers):
        self._headers = headers

    def get_news_urls(self, query="bts", start=1, display=100) -> list:
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

    def read_news(self, url: str) -> dict:
        res = req.get(url)
        if res.status_code == 200:
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
        else:
            raise Exception(f"Response failed. Code: {res.status_code}")

    def __call__(self, query, n):
        urls = self.get_news_urls(query=query, display=n)
        news = [self.read_news(url) for url in urls]
        # news = self.read_news(urls[0])

        return news
