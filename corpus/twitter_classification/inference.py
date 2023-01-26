import argparse

import numpy as np
import pandas as pd
import torch
from dataset import TwitterClassificationDataset
from omegaconf import OmegaConf
from transformers import ElectraForSequenceClassification, ElectraTokenizer, Trainer


def inference(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    model = ElectraForSequenceClassification.from_pretrained(
        config.model.name_or_path,
        num_labels=3,
        problem_type="multi_label_classification",
    )
    tokenizer = ElectraTokenizer.from_pretrained(config.model.name_or_path)

    # load csv
    df = pd.read_csv(config.data.pickle_to_csv_path)

    # split train and eval stratified
    df.reset_index(drop=True, inplace=True)
    df["label"] = [999 for _ in range(len(df))]

    # load dataset
    test_dataset = TwitterClassificationDataset(df, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    # inference
    predictions = trainer.predict(test_dataset)

    # save predictions to csv
    df["pred"] = np.argmax(predictions.predictions, axis=1)
    df.to_csv(config.data.inference_save_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="tweet_classification_config")

    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./data/twitter_data_preprocess/{args.config}.yaml")

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    inference(config)
