from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer

ckpt_path = "./saved_models/StoneSeller/roberta-large-ssm-wiki-e2/mrc_None_01-05-02-01/checkpoint-21785/"
save_name = "nlpotato/roberta_large-ssm_wiki_e2-origin_added_korquad_e5"

model = AutoModel.from_pretrained(ckpt_path)
model.push_to_hub(save_name)
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
tokenizer.push_to_hub(save_name)