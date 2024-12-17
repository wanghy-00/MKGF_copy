import json
import os
import sys
sys.path.append('.')
import random
import argparse
import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

from PIL import Image

class cl_dataset(Dataset):
    def __init__(self, ent2img, cl_dict, neg_num, image_processor):
        self.ent2img = ent2img
        self.cl_dict = cl_dict
        self.query_entities = list(cl_dict)

        self.neg_num = neg_num
        self.image_processor = image_processor

    def __len__(self):
        return len(self.query_entities)

    def __getitem__(self, index):
        query_entity = self.query_entities[index]
        if len(self.cl_dict[query_entity]) < self.neg_num:
            neg_entities = np.random.choice(self.cl_dict[query_entity], size=self.neg_num, replace=True)
        else:
            neg_entities = self.cl_dict[query_entity][:self.neg_num]
        
        img_path = "/root/nas/image-crawler/images/slake"
        query_image_input = self.image_processor(Image.open(os.path.join(img_path, query_entity, self.ent2img[query_entity][0])))
        positive_image_input = self.image_processor(Image.open(os.path.join(img_path, query_entity, self.ent2img[query_entity][1])))
        negative_image_inputs = torch.stack([self.image_processor(Image.open(os.path.join(img_path, ent, self.ent2img[ent][0]))) for ent in neg_entities])

        return query_image_input, positive_image_input, negative_image_inputs


class RetrieverDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor, fold="train"):
        self.args = args

        with open(os.path.join(args.data_dir, "train.json"), 'r') as f:
            self.dataset = json.load(f)
        self.image_processor = image_processor
    
        with open("/root/nas/QA/Med-QA/Reranker/dataset/slake/e2image.json", 'r') as f:
            self.e2img = json.load(f)

        if args.debug:
            self.original_dataset = self.original_dataset[:100]

        self.dataset = [(_id, data) for _id, data in enumerate(self.dataset)]

        self.tokenizer = tokenizer
        self.max_seq_len = 512

        self.fold = fold
        self.k = args.n_cands if fold == "train" else 8
        self.n_k = 32
        
        os.makedirs(args.search_space_dir, exist_ok=True)
        self.search_space_dump = args.search_space_dir

    def preprocess(self, data, space, _idx):
        img_path = "/root/nas/image-crawler/images/slake"
        query = data["question"]

        query_image = Image.open(os.path.join("/root/nas/QA/Med-QA/SLAKE/imgs", data["img_name"]))
        query_image_input = self.image_processor(query_image)

        keys = [item["chunk"] for item in space]

        query_outputs = self.tokenizer(query, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)
        key_outputs = self.tokenizer(keys, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)

        key_images = [Image.open(os.path.join("/root/nas/image-crawler/images/slake", I["entity"], self.e2img[I["entity"]])) for I in space]
        key_image_inputs = torch.stack([self.image_processor(key_image) for key_image in key_images])
        
        scores = [item["score"] - 0.8 * index/100 for index, item in enumerate(space)]

        query_input_ids = query_outputs["input_ids"]
        query_attention_mask = query_outputs["attention_mask"]
        key_input_ids = key_outputs["input_ids"]
        key_attention_mask = key_outputs["attention_mask"]
        score = torch.FloatTensor(scores)

        return query_input_ids, query_attention_mask, query_image_input, key_input_ids, key_attention_mask, key_image_inputs, score


    def __getitem__(self, idx):
        _idx, data = self.dataset[idx]
        load_path = os.path.join(self.search_space_dump, str(_idx).zfill(4) + ".json") # score?
        with open(load_path, 'r') as f:
            _ss = json.load(f)
        
        ss = _ss
        return self.preprocess(data, ss, _idx)

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../Biomedical_CoT_Generation/data/usmle-cot")
    parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--n_negative", type=int, default=6, help="The number of negative samples")
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    RetrieverDataset(args, tokenizer)