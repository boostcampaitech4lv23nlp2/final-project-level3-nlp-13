import pandas as pd
import torch


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        raw_data = pd.read_csv(file_path)
        train_data = "<s>" + raw_data["Q"] + "</s>" + "<s>" + raw_data["A"] + "</s>"
        # <s>안녕하세요</s><s> -> 네, 안녕하세요</s>
        return self.tokenizer(list(train_data), padding="max_length", max_length=self.max_len, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        return (self.data["input_ids"][index], self.data["attention_mask"][index], self.data["token_type_ids"][index])
