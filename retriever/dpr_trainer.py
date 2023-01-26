import json
import logging
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from dpr_dataset import DPRDataset
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DPRTrainer:
    def __init__(self, args, config, tokenizer, p_encoder, q_encoder, train_dataset, db_dataset, train_data, valid_data):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.train_dataset = train_dataset
        self.db_dataset = db_dataset
        self.valid_questions = valid_data["Q"]
        self.valid_answers = valid_data["A"]
        self.db_questions_data = train_data["Q"] + valid_data["Q"]
        self.db_answers_data = train_data["A"] + valid_data["A"]

        # logger 설정
        if not os.path.exists("./retriever/logs"):
            os.makedirs("./retriever/logs")
            with open("./retriever/logs/dpr_log.log", "w+") as f:
                f.write("***** Make Log File *****\n")
        LOG_FORMAT = "%(asctime)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
        file_handler = logging.FileHandler("./retriever/logs/dpr_log.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    def configure_optimizer(self, optimizer_grouped_parameters, config):
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.trainer.learning_rate)
        return optimizer

    def train_per_epoch(self, epoch_iterator: DataLoader, optimizer):
        batch_loss = 0

        for step, batch in enumerate(epoch_iterator):
            self.p_encoder.train()
            self.q_encoder.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0].squeeze(1),
                "attention_mask": batch[1].squeeze(1),
                "token_type_ids": batch[2].squeeze(1),
            }

            q_inputs = {
                "input_ids": batch[3].squeeze(1),
                "attention_mask": batch[4].squeeze(1),
                "token_type_ids": batch[5].squeeze(1),
            }

            p_outputs = self.p_encoder(**p_inputs)
            q_outputs = self.q_encoder(**q_inputs)

            # calculate similarity score
            sim_scores = torch.matmul(p_outputs, q_outputs.transpose(0, 1))  # (batch_size, batch_size)

            # target = position of positive sample = diagonal
            target = torch.arange(0, self.args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                target = target.to("cuda")

            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, target)
            loss.backward()
            optimizer.step()
            self.q_encoder.zero_grad()
            self.p_encoder.zero_grad()

            batch_loss += loss.detach().cpu().numpy()
        torch.cuda.empty_cache()
        return batch_loss / len(epoch_iterator)

    def valid_per_epoch(self, db_dataloader, epoch):
        logger.info("***** Running Validation *****")

        # passage embedding 생성
        p_embs = []

        with torch.no_grad():
            epoch_iterator = tqdm(db_dataloader, desc="Iteration")
            self.p_encoder.eval()

            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)
                p_inputs = {
                    "input_ids": batch[0].squeeze(1),
                    "attention_mask": batch[1].squeeze(1),
                    "token_type_ids": batch[2].squeeze(1),
                }
                outputs = self.p_encoder(**p_inputs).to("cpu").numpy()
                p_embs.extend(outputs)

            # p_embs = np.array(p_embs)
            p_embs = torch.Tensor(p_embs)  # (num_of_passages, 768)

        if not os.path.exists("./retriever/saved_models/dpr/passage_embs"):
            os.makedirs("./retriever/saved_models/dpr/passage_embs")
        with open(f"./retriever/saved_models/dpr/passage_embs/p_embs_{epoch+1}.bin", "wb") as f:
            pickle.dump(p_embs, f)

        # question embedding 생성
        top_1, top_3, top_5 = 0, 0, 0

        with torch.no_grad():
            self.q_encoder.eval()

            for idx in tqdm(range(len(self.valid_questions))):
                q_seq = self.tokenizer(self.valid_questions[idx], max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(
                    "cuda"
                )
                q_emb = self.q_encoder(**q_seq).to("cpu")  # (1, 768)

                # calculate similarity score
                sim_scores = torch.matmul(q_emb, p_embs.transpose(0, 1))  # (1, num_passage)
                rank = torch.argsort(sim_scores, dim=1, descending=True)

                top_1_passages = [self.db_answers_data[i] for i in rank[0][:1]]
                top_3_passages = [self.db_answers_data[i] for i in rank[0][:3]]
                top_5_passages = [self.db_answers_data[i] for i in rank[0][:5]]

                # top_k accuracy
                if self.valid_answers[idx] in top_1_passages:
                    top_1 += 1
                if self.valid_answers[idx] in top_3_passages:
                    top_3 += 1
                if self.valid_answers[idx] in top_5_passages:
                    top_5 += 1

        return (
            top_1 / len(self.valid_questions) * 100,
            top_3 / len(self.valid_questions) * 100,
            top_5 / len(self.valid_questions) * 100,
        )

    def train(self):
        logger.info("***** Running Training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Batch size per device = %d", self.args.per_device_train_batch_size)

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.per_device_train_batch_size, drop_last=True)
        db_dataloader = DataLoader(self.db_dataset, batch_size=self.args.per_device_eval_batch_size)

        best_top_1, best_top_3, best_top_5 = 0, 0, 0

        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = self.configure_optimizer(optimizer_grouped_parameters, self.config)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        warmup_step = int(t_total * self.args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        # train
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            # train per epoch
            train_loss = self.train_per_epoch(epoch_iterator, optimizer)
            logger.info("Train loss: {}".format(train_loss))

            # validation per epoch
            top_1, top_3, top_5 = self.valid_per_epoch(db_dataloader, epoch)

            logger.info("***** Validation Result *****")
            logger.info(f"epoch: {epoch+1} | train_loss: {train_loss} | top_1: {top_1} | top_3: {top_3} | top_5: {top_5}")

            scheduler.step()

            if top_1 > best_top_1:
                best_top_1 = top_1
                self.q_encoder.save_pretrained("./retriever/saved_models/dpr/q_encoder/q_encoder_best_top_1")
                self.p_encoder.save_pretrained("./retriever/saved_models/dpr/p_encoder/p_encoder_best_top_1")
            if top_3 > best_top_3:
                best_top_3 = top_3
                self.q_encoder.save_pretrained("./retriever/saved_models/dpr/q_encoder/q_encoder_best_top_3")
                self.p_encoder.save_pretrained("./retriever/saved_models/dpr/p_encoder/p_encoder_best_top_3")
            if top_5 > best_top_5:
                best_top_5 = top_5
                self.q_encoder.save_pretrained("./retriever/saved_models/dpr/q_encoder/q_encoder_best_top_5")
                self.p_encoder.save_pretrained("./retriever/saved_models/dpr/p_encoder/p_encoder_best_top_5")
