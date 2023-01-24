### 트위터 사용 방법
1. `.env` 파일에 개인 계정 key 및 token을 작성 (`.env.template` 참고)
2. 크롤링하고자 하는 user의 screen_name(@은제외)을 아래와 같이 입력
3. `python build_corpus.py -c=twitter -s=<screen_name>`


### 네이버 사용 방법
`python build_cropus.py -c=naver -n=100 -q=BTS`
## Args
* n : 크롤링할 기사 수
* q : 검색할 스트링


### 더쿠 사용 방법
1. chromedriver 설치 필수.
2. 크롤링하고자 하는 페이지의 수를 입력. 한 페이지당 20개의 링크.
3. `python build_corpus.py -c=theqoo -n=100
