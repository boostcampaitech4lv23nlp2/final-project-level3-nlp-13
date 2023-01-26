from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from tqdm import tqdm
from soynlp.normalizer import *
import pickle
import time
import os
import glob
import re


chrome_options=Options()

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')



class TheqooCrawler:
    def __init__(self):
        self.driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', chrome_options=chrome_options)
        self.save_path = "data/raw_data/theqoo"
        self.pages = None


    def get_urls(self, pages: list):
        """ 페이지 리시트에서 게시글(20개) * 크롤링 할 페이지 개수 -> url의 리스트를 반환
        Args:
            pages (list) : 2면 [2,3] 항상 2부터시작(1부터는 1시간내라 안읽힘), 3이면 [2,3,4]
        """
        urls = []
        for page in pages:
            page = 'https://theqoo.net/index.php?mid=bts&filter_mode=best&m=0&page=' + str(pages)
            self.driver.get(page)
            rows = self.driver.find_elements(By.TAG_NAME, 'a')
        
            for row in rows:
                url = row.get_attribute('href')
                if url and ('document_srl=') in url and not url.endswith('comment'):
                    urls.append(url)

        return urls
    
    def check_text(self, text: str):
        """ 크롤링한 글의 내용이 적합한지 확인
        Args:
            text (str) : 크롤링한 글의 내용
        """
        if 'http' not in text and '://' not in text and '.com' not in text:
            return True
        else:
            return False

    def check_comment(self, comment: str):
        """ 크롤링한 댓글에서 필요없는부분은 삭제
        Args:
            comment (str) : 크롤링한 댓글 중 1개
        """
        if '로그인 후에 바로 열람 가능합니다' not in comment and '무명의 더쿠' not in comment and '삭제된 댓글입니다' not in comment and '비회원은 작성한 지 1시간 이내의' not in comment and self.check_text(comment):
            return True
        else:
            return False


    def preprocess(self, comment: str):
        """개행 문자, URL, @id, 앞뒤 공백 제거
        Args:
            sent (str): 전처리할 문장
        Returns:
            sent (str): 전처리된 문장
        """
        comment = comment.replace("\n", " ")
        comment = comment.replace("  ", " ")
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
        comment = emoji_pattern.sub(r'', comment) # no emoji
        comment = re.sub(r"@\S+", "", comment)
        comment = comment.lstrip()
        comment = comment.rstrip()

        comment = emoticon_normalize(comment, num_repeats=2)
        comment = repeat_normalize(comment, num_repeats=2)
        comment = only_text(comment)

        if '덬' in comment or 'MB' in comment or 'GB' in comment or '☞' in comment or '.zip' in comment or '대용량 파일' in comment or '다운로드 가능' in comment:
            comment = ''

        return comment

    def get_data(self, url: str):
        """url을 받아서 text(글내용), comments(댓글내용)의 dict를 반환
        Args:
            url (str): 하나의 url 링크
        """
        dict = {}
        print("***** 크롤링을 시작합니다 *****")
        self.driver.get(url)
        xpath_button = '//*[@id="cmtPosition"]/div[2]'

        while self.driver.find_element(By.XPATH, xpath_button).is_displayed:
            try:
                WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, xpath_button))
                ).click() 
                time.sleep(1)
            except TimeoutException:
                break

        text = self.driver.find_element(By.TAG_NAME, 'article').text
        if self.check_text(text): # text가 url을 포함하지 않을때만 크롤링
            comments_set = set()
            comments_element = self.driver.find_elements(By.CLASS_NAME, 'fdb_lst_ul')
            for com in comments_element:
                for sentence in list(com.text.split('\n')):
                        if self.check_comment(sentence):
                            sentence = self.preprocess(sentence)
                            if sentence:
                                comments_set.add(sentence)

            text = self.preprocess(text)
            dict['text'] = text
            dict['comments'] = list(comments_set)

        return dict

    def check_filepath(self, save_path: str):
        """데이터 저장 경로 폴더가 존재하는지 확인하고 없다면 생성
        Args:
            save_path (str): 저장 경로
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)


    def __call__(self, n: int):
        self.check_filepath(self.save_path)
        self.pages = [i+2 for i in range(n)] # 크롤링은 항상 2페이지부터 시작.
        self.screen_name = 'bts_hot'

        self.urls = self.get_urls(self.pages) # 한 페이지당 20개의 url
        
        result = [] 
        for url in tqdm(self.urls):
            result_data = self.get_data(url)
            if len(result_data) and len(result_data['text']) > 0:
                result.append(result_data)
        
        file_name = f"theqoo_{self.screen_name}.pickle"
        pickle_files = ""
        if glob.glob(f"{self.save_path}/*.pickle"):
            pickle_files = glob.glob(f"{self.save_path}/*.pickle")

        if pickle_files and f"{self.save_path}/{file_name}" in pickle_files:
            print(f"***** update {file_name} ******")
            with open(f"{self.save_path}/{file_name}", "rb") as f:
                data = pickle.load(f)
                data.update(result)
            with open(f"{self.save_path}/{file_name}", "wb") as f:
                pickle.dump(data, f)
        else:
            print(f"***** make {file_name} *****")
            with open(f"{self.save_path}/{file_name}", "wb") as f:
                pickle.dump(result, f)