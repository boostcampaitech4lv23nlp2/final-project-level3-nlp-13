# BTS를 덕질하는 트위터 챗봇, ArmyBot 

## 1️⃣ Introduction
<img src="https://user-images.githubusercontent.com/65378914/217187454-b8159fff-7152-4125-9a18-2c0ccf236aeb.png"  width="700" height="180"/>
같이 덕질하자! ArmyBot은 흔한 아미 트친(BTS을 응원하는 트위터 친구)처럼 이런 저런 이야기를 나눌 수 있는 챗봇 서비스입니다. BTS 관련 덕심 가득한 질문부터 일상 대화까지 [@armybot_13](https://twitter.com/armybot_13)으로 트윗만 보내면 ArmyBot이 답장을 합니다.

## 2️⃣ 팀원 소개

김별희|이원재|이정아|임성근|정준녕|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/42535803?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/61496071?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/65378914?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/14817039?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/51015187?v=4' height=80 width=80px></img>
[Github](https://github.com/kimbyeolhee)|[Github](https://github.com/wjlee-ling)|[Github](https://github.com/jjeongah)|[Github](https://github.com/lim4349)|[Github](https://github.com/ezez-refer)

## Contribution

- `김별희` 트위터 데이터 수집 및 전처리 파이프라인 구축, answer retriever 구축
- `이원재` 서비스 기획 및 PM, 사전학습용 데이터 수집 및 전처리, 답변 관련 처리
- `이정아` 스팸 필터링 데이터 수집 및 모델 구축, 트위터 연결 및 서비스
- `임성근` 더쿠 데이터 수집 및 전처리 파이프라인 구축, 정보성 데이터 수집
- `정준녕` 생성 모델 구축, 생성 모델 학습용 데이터 수집, 프로토타입 구현

## 3️⃣ Demo Video
![service example](https://im.ezgif.com/tmp/ezgif-1-a031a0f781.gif)

## 4️⃣ Service Architecture
![model](https://user-images.githubusercontent.com/65378914/217187792-2f0c2f93-d389-46d9-b361-d8ac7e4434c6.png)
### Project Tree

```
.
|-- chatbot
|   `-- generator
|       |-- config
|       |   `-- base_config.yaml
|       |-- data_loader
|       |   `-- data_loaders.py
|       |-- train.py
|       `-- trainer
|           `-- trainer.py
|-- config
|   `-- base_config.yaml
|-- corpus
|   |-- build_corpus.py
|   `-- crawlers
|       |-- __init__.py
|       |-- aihub_crawler.py
|       |-- kin_crawler.py
|       |-- naver_crawler.py
|       |-- naver_jisikin_crawler.py
|       |-- theqoo_crawler.py
|       `-- twitter_crawler.py
|-- inference.py
|-- install_requirements.sh
|-- notebook
|   |-- AIhub_data_to_csv.ipynb
|   `-- upload_dataset_to_huggingface.ipynb
|-- poetry.lock
|-- pretraining
|   |-- run_bart_dlm_flax.py
|   |-- run_clm_flax.py
|   `-- run_t5_mlm_flax.py
|-- prototype
|   |-- Makefile
|   |-- app
|   |   |-- __main__.py
|   |   |-- frontend.py
|   |   |-- main.py
|   |   |-- model.py
|   |   `-- pyproject.toml
|   |-- config
|   |   `-- base_config.yaml
|   |-- poetry.lock
|   |-- pyproject.toml
|   `-- requirements.txt
|-- pyproject.toml
|-- readme.md
|-- requirements.txt
`-- utils
    |-- EDA.py
    |-- push_model_to_hub.py
    `-- util.py
```

## 5️⃣ DataSet

![dataset](https://user-images.githubusercontent.com/65378914/217187965-e6cf8c16-38af-4392-bdd0-63bcb6f362fb.png)
- [AI Hub 연예뉴스](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=625) : 3,144개 10.67MB
- 네이버 뉴스 BTS 관련 기사 : 1,337개 4.85MB
- [일상 대화 및 위로 문답 챗봇 데이터](https://github.com/songys/Chatbot_data) : 962,681개 108.43MB
- BTS 관련 네이버 지식인 : 7,785개 8.70MB
- 더쿠 BTS 카테고리 글/댓글 : 13,709개 3.53MB
- 트위터 BTS 팬 트윗/답글 : 8,106개 1.45MB
- [Korean-hate-speech](https://github.com/kocohub/korean-hate-speech) : 7,896개
- [KOLD](https://github.com/boychaboy/KOLD) : 40,429개
- [Korean_unsmile_data](https://github.com/smilegate-ai/korean_unsmile_dataset) : 7,896개
- [Curse-detection-data](https://github.com/2runo/Curse-detection-data) : 6,154개

## 6️⃣ Modeling

## 7️⃣ How to Run
### Clone Repo & Install dependency

```python
$ git clone https://github.com/boostcampaitech4lv23nlp2/final-project-level3-nlp-13.git
$ cd final-project-level3-nlp-13
$ poetry install
```

### Set up Elastic Search

```python
$ bash install_elastic_search.sh
```

### Run

```python
$ python agent.py
```
## 8️⃣  Future Works
- 생성모델 성능 개선
- FastText를 이용해 임베딩
- Salient Span Masking을 도입한 사전학습
- 답장 외 챗봇의 글 생성 기능 및 이벤트 기능 추가

## 9️⃣ Development Environment

- 협업툴 : Notion, Slack, Huggingface, Wandb
- 개발 환경
    - GPU: V100
    - 언어: Python==3.8.5
    - dependency: PyTorch == 1.13.1
