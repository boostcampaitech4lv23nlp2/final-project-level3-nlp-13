from datasets import load_dataset
from dpr_dataset import DPRDataset
from transformers import BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

    # load data from huggingface dataset
    data = load_dataset("nlpotato/chatbot_twitter_ver2")
    train_data = data["train"]
    valid_data = data["test"]

    # dataset
    train_dataset = DPRDataset(train_data["Q"], train_data["A"], tokenizer)
    valid_dataset = DPRDataset(valid_data["Q"], valid_data["A"], tokenizer)

    print(len(train_dataset))
