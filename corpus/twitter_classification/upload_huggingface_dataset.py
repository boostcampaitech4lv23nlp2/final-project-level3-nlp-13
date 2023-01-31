import os

import pandas as pd
from datasets import load_dataset

if __name__ == "__main__":
    # final 폴더에 있는 모든 파일을 읽어서 데이터 프레임으로 만들기
    df = pd.DataFrame(columns=["Q", "A"])
    for file in os.listdir("./data/processed_data/twitter/final"):
        if file.endswith(".csv"):
            print(file)
            df = pd.concat([df, pd.read_csv(f"./data/processed_data/twitter/final/{file}")])

    # 데이터 셔플
    df = df.sample(frac=1).reset_index(drop=True)
    # 데이터프레임 train valid로 분리
    train_df, valid_df = df[: int(len(df) * 0.9)], df[int(len(df) * 0.9) :]

    # huggingface dataset으로 변환하기 위해 json 형식으로 변환
    if not os.path.exists("./data/processed_data/twitter"):
        os.makedirs("./data/processed_data/twitter")
    train_df.reset_index().to_json("./data/processed_data/twitter/twitter_train.json", orient="records")
    valid_df.reset_index().to_json("./data/processed_data/twitter/twitter_valid.json", orient="records")

    # huggingface dataset으로 변환
    data_files = {
        "train": "./data/processed_data/twitter/twitter_train.json",
        "test": "./data/processed_data/twitter/twitter_valid.json",
    }
    huggingface_dataset = load_dataset("json", data_files=data_files)

    huggingface_dataset.push_to_hub(repo_id="nlpotato/chatbot_twitter_ver3_temp", private=False)

    print(huggingface_dataset)
