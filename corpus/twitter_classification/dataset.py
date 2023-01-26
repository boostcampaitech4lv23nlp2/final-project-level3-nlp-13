import torch
from torch.utils.data import Dataset


class TwitterClassificationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        Q = self.data.iloc[index, 0]
        A = self.data.iloc[index, 1]
        text = str(Q) + "[SEP]" + str(A)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        label = None
        if self.data.iloc[index, 2] == 0:
            label = [1, 0, 0]
        elif self.data.iloc[index, 2] == 1:
            label = [0, 1, 0]
        elif self.data.iloc[index, 2] == 2:
            label = [0, 0, 1]

        if label:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.float),
            }
        else:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
            }
