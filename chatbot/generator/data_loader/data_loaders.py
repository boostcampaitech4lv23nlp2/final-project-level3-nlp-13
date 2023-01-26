import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from datasets import Dataset, DatasetDict, load_dataset


class GPT_Dataset:
    def __init__(self, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_len = self.config.tokenizer.max_length
        self.raw_datasets = load_dataset(self.config.path.data)
        self.tokenized_datasets = self.raw_datasets.map(
            self.tokenize,
            batched=True,
            remove_columns=self.raw_datasets["train"].column_names,
        )

    def tokenize(self, element):
        outputs = self.tokenizer(
            list(pd.DataFrame({"Q": element["Q"], "A": element["A"]}).apply(lambda x: "</s>" + x["Q"] + "<sep>" + x["A"] + "</s>", axis=1)),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_length=True,
        )

        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == self.max_len:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}


class Enc_Dec_Dataset:
    def __init__(self, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_len = self.config.tokenizer.max_length
        self.raw_datasets = load_dataset(self.config.path.data)
        self.tokenized_datasets = self.raw_datasets.map(
            self.tokenize,
            batched=True,
            remove_columns=self.raw_datasets["train"].column_names,
        )

    def tokenize(self, element):
        inputs = self.tokenizer(
            element["Q"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
        )

        target_tokens = self.tokenizer(
            element["A"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=False,
        )["input_ids"]

        return {**inputs, "labels": target_tokens}
