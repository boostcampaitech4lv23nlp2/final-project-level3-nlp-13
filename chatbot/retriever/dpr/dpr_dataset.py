import torch
from torch.utils.data import Dataset


class DPRDataset(Dataset):
    def __init__(self, queries, contexts, tokenizer):
        self.queries = queries
        self.contexts = contexts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        context = self.contexts[idx]

        query_tensor = self.tokenize(query)
        context_tensor = self.tokenize(context)

        return (
            query_tensor["input_ids"],
            query_tensor["attention_mask"],
            query_tensor["token_type_ids"],
            context_tensor["input_ids"],
            context_tensor["attention_mask"],
            context_tensor["token_type_ids"],
        )

    def tokenize(self, input):
        return self.tokenizer(
            input,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
