import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from datasets import Dataset, DatasetDict,load_dataset

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        try:
            self.data = self.load_data(file_path)
        except:
            self.data = load_dataset(file_path)

    def load_data(self, file_path):
        raw_data = pd.read_csv(file_path)
        train_data = "<s>" + raw_data["Q"] + "</s>" + "<s>" + raw_data["A"] + "</s>"
        # <s>안녕하세요</s><s> -> 네, 안녕하세요</s>
        return self.tokenizer(list(train_data), padding="max_length", max_length=self.max_len, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        return (self.data["input_ids"][index], self.data["attention_mask"][index], self.data["token_type_ids"][index])


class GPTDataset:
    def __init__(self, tokenizer, file_path, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        try:
            self.raw_datasets = self.load_data(file_path)
        except:
            self.raw_datasets = load_dataset(file_path)
        self.tokenized_datasets = self.raw_datasets.map(
            self.tokenize,
            batched=True,
            remove_columns=self.raw_datasets["train"].column_names,
        )

    def load_data(self, file_path):
        raw_data = pd.read_csv(file_path)
        raw_datasets = DatasetDict(
            {
                "train": Dataset(pa.Table.from_pandas(raw_data)),  # .shuffle().select(range(50000)),
                "valid": Dataset(pa.Table.from_pandas(raw_data)),  # .shuffle().select(range(50000)),
            }
        )
        return raw_datasets

    def tokenize(self, element):
        outputs = self.tokenizer(
            list(pd.DataFrame({"Q": element["Q"], "A": element["A"]}).apply(lambda x: "</s>" + x["Q"] + "<sep>" + x["A"] + "</s>", axis=1)),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_overflowing_tokens=True,
            return_length=True,
        )

        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == self.max_len:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}


class SingleSentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
