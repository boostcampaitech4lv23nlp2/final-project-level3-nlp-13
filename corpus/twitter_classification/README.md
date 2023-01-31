# How to Train
1. config의 data.train_data_path 경로에 학습에 사용할 라벨링 데이터 경로 입력
2. `python corpus/twitter_classification/train.py` 


# How to Inference
## 경로 안에 있는 모든 pickle 파일을 Inference해서 사용할 데이터만 추출
1. config의 data.witter_pickle_path 경로에 pickle 파일들의 경로를 입력
2. config의 data.pickle_to_csv_path 경로에 `pickle 파일들을 csv 형태로 변환`한 파일 저장 경로 입력
2. config의 inference_save_path 경로에 `prediction 된 label 값이 있는 파일` 저장할 경로 입력
3. config의 final_save_path 경로에는 `사용할 데이터만 있는 파일` 저장할 경로 입력
4. config의 model.name_or_path에 학습된 모델 경로 입력
5. python corpus/twitter_classification/data_process.py

