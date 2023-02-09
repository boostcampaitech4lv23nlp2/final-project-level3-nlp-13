## corpus 폴더 역할 및 env 파일 
데이터를 수집, 전처리하여 코퍼스를 구축하는 서브모듈. 수집 및 전처리는 build_corpus.py로 실행할 수 있습니다.<br>
.env.template 파일을 복사해 twitter, Naver API 로그인 정보를 입력 후 .env 파일로 저장해야 합니다.

## Twitter 
크롤링하고 싶은 계정을 입력합니다.

#### 크롤링
```
python build_corpus.py -c twitter -s {screen_name} --do_crawl
```

#### 전처리
```
corpus/twitter_classifcation의 README 참고
```

## Naver News
주어진 키워드에 관한 네이버 연예뉴스, 일반뉴스를 수집합니다. 네이버가 1차적으로 토픽별로 클러스터링한 기사들 중 대표 기사들만 수집합니다.

#### 크롤링
```
python build_corpus.py -c naver -q {키워드} -n {수집량} --do_crawl
```

#### 전처리
```
python build_corpus.py -c naver -p {전처리할 파일이나 폴더} --do_preprocess
```

## Naver 지식인(kin)
주어진 키워드에 관한 네이버 지식인 문답을 수집합니다. 네이버 검색 API ID와 비밀번호가 있어야 수집이 가능하며, ID와 비밀번호는 .env 파일에 저장되어 있어야 합니다.

#### 크롤링
```
python build_corpus.py -c kin -q {키워드} -n {수집량} --do_crawl
```

## 더쿠
주어진 키워드에 관한 더쿠 hot게시판, 방탄 게시판 ~ 2페이지부터 수집.(1페이지는 비회원 시간제한있음), 1페이지당 세부 20개의 링크가있어서 n * 20개의 사이트에서 글 내용과 댓글을 수집합니다.

#### 크롤링
```
python build_corpus.py -c theqoo -n {수집량}
```
