import pickle
import os
import re 
import time
from typing import List, Union

import requests as req
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import trange
from dotenv import load_dotenv

load_dotenv()
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")


class NaverCrawler:
    def __init__(self, runtime: str):
        self._headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        }

        self.save_path = "data/raw_data/naver"
        self.runtime = runtime
        self.driver = webdriver.Chrome("./chromedriver")

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
            raise Exception(f"Failed to get response from Naver API. Code: {res.status_code}")  # TO-DO: use logger

        return urls

    def read_news(self, url: str) -> Union[dict, None]:
        try:
            self.driver.get(url)
            self.driver.implicitly_wait(10)
            parsed = self.parse()            
            time.sleep(3.2)
            return parsed # TO-DO: class or namedtupel for news results
        except:
            return None

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
            url = urls[idx]
            news = self.read_news(url)
            if isinstance(news, dict):
                item = {"id": f"naver_{query}_{idx}", "url": url}
                item.update(news)
                output["data"].append(item)

        self.driver.quit()
        print(f"Crawled {len(output['data'])} articles from the given query '{query}'")  # TO-DO: change to logger
        self.save(query=query, run_time=self.runtime, data=output)

    def parse(self) -> dict:
        """
        기사(뉴스 또는 연예뉴스)에 따라 다른 tag을 갖고 있기 때문에 달리 parse
        """
        by = By.CSS_SELECTOR
        try:
            # entertain news
            title = self.driver.find_element(by, "h2.end_tit").text.strip()
            body = self.driver.find_element(by, "div.article_body").text.strip()  # class: article_body
            if caps:= self.driver.find_elements(by, "em.img_desc"):
                img_captions = [cap.text.strip() for cap in caps]
            written_at = self.driver.find_element(by, "span > em").text.strip()
            writer = self.driver.find_element(by, "p.byline_p > span").text.strip()

        except:
            # news
            title = self.driver.find_element(by, "h2.media_end_head_headline").text.strip()
            body = self.driver.find_element(by, "div._article_content").text.strip()
            if caps:= self.driver.find_elements(by, "em.img_desc"):
                img_captions = [cap.text.strip() for cap in caps]
            written_at = self.driver.find_element(by, "span._ARTICLE_DATE_TIME").text.strip()
            writer = self.driver.find_element(by, "em.media_end_head_journalist_name").text.strip()

        parsed = {
            "title": title,
            "body": body,
            "written_at": written_at,
            "writer": writer,
            "caption": img_captions,
        }
        return parsed

    def preprocess(self, title, body, img_captions, written_at, writer):
        body = self.remove_caption(body, img_captions)

        return title, body, written_at, writer

    def remove_caption(self, text:str, captions:List) -> str:
        """
        기사 본문 text에 포함된 이미지 caption을 제거함.
        """
        for caption in captions:
            pattern = re.compile(caption)
            text = re.sub(pattern, "", text)
        return text

    def save(self, query: str, run_time: str, data: dict) -> None:
        size = len(data["data"])
        with open(f"{self.save_path}_{query}_{run_time}_size_{size}.pickle", "wb") as f:
            pickle.dump(data, f)
            print(f"Saved to {self.save_path}")
