from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer, BertForSequenceClassification

ckpt_path = "./spam_saved_models/klue/bert-base_01-19-11-18_10epoch"
save_name = "nlpotato/spam-small-data-bert-base-10e"

model = BertForSequenceClassification.from_pretrained(ckpt_path)
model.push_to_hub(save_name)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
tokenizer.push_to_hub(save_name)