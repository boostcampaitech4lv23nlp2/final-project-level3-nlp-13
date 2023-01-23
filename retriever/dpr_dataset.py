from torch.utils.data import Dataset


class DPRDataset(Dataset):
    def __init__(self, questions, answers, tokenizer):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        question_encoding = self.tokenize(question)
        answer_encoding = self.tokenize(answer)

        return {
            "question": {
                "input_ids": question_encoding["input_ids"].flatten(),
                "attention_mask": question_encoding["attention_mask"].flatten(),
                "token_type_ids": question_encoding["token_type_ids"].flatten(),
            },
            "answer": {
                "input_ids": answer_encoding["input_ids"].flatten(),
                "attention_mask": answer_encoding["attention_mask"].flatten(),
                "token_type_ids": answer_encoding["token_type_ids"].flatten(),
            },
        }

    def tokenize(self, input):
        encoding = self.tokenizer(input, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

        return encoding
