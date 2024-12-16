from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch
import numpy as np
from einops import rearrange

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from transformers import AutoTokenizer

from torch import nn
import torch.nn.functional as F

# class ColBERTConfig(PretrainedConfig):
#     compression_dim: int = 768
#     dropout: float = 0.0
#     return_vecs: bool = False
#     trainable: bool = True

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        计算对比损失
        :param anchor: [batch_size, feature_dim] 给定样本（锚点样本）
        :param positive: [batch_size, feature_dim] 正样本
        :param negatives: [batch_size, num_negatives, feature_dim] 负样本集合
        :return: 对比损失
        """
        device = anchor.device

        # 标准化特征向量
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=2)

        # 计算正样本与锚点样本之间的相似度
        positive_similarity = torch.sum(anchor * positive, dim=1) / self.temperature
        # print("positive_similarity:", positive_similarity)

        # 计算负样本与锚点样本之间的相似度
        negative_similarity = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature
        # print("negative_similarity:", negative_similarity)

        # 计算对比损失
        logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1)
        # print("logits:", logits)
        labels = torch.zeros(anchor.size(0)).to(device).long()  # 正样本的标签为 0  # 4个N+1分类问题，N+1分类的对应的类别标签为第一个正例，即下标为0
        # print("labels:", labels)
        
        loss = F.cross_entropy(logits, labels)
        return loss



class ColBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.BioCLIP, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.text_linear = nn.Sequential(nn.Linear(512, 512),
                                        nn.Tanh(),
                                        nn.Linear(512, 512))

        self.image_linear = nn.Sequential(nn.Linear(512, 512),
                                        nn.Tanh(),
                                        nn.Linear(512, 512))


        self.criterion = NTXentLoss(temperature=0.5)
    

    def forward(self,
                queries: Dict[str, torch.LongTensor],
                query_image_input,
                documents: Dict[str, torch.LongTensor], 
                key_image_inputs,
                device):
        scores = []
        # query_vecs = self.BioCLIP.text(query)
        # print(query_vecs.shape)

        # logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1) # BioCLIP样例 图文对的算法，本文是图—图对，文文对算相似度

        for query, query_image, document, key_image in zip(queries, query_image_input, documents, key_image_inputs):
            # query = query.unsqueeze(0)
            query_image = query_image.unsqueeze(0)

            # print("query:", query.shape)
            # print("query_image:", query_image.shape)

            query_vecs = self.BioCLIP.text(query)
            query_vecs= self.text_linear(query_vecs)
            query_img_vecs = self.BioCLIP.visual(query_image)
            query_img_vecs= self.image_linear(query_img_vecs)

            # print("query_vecs:", query_vecs.shape)

            # print("document:", document.shape)
            # print("key_image:", key_image.shape)
            document_vecs = self.BioCLIP.text(document)
            document_vecs= self.text_linear(document_vecs)
            document_img_vecs = self.BioCLIP.visual(key_image)
            document_img_vecs = self.image_linear(document_img_vecs)
            # print("document_vecs:",document_vecs.shape)
            

            # text_score = F.cosine_similarity(query_vecs, document_vecs, dim=-1)
            # image_score = F.cosine_similarity(query_img_vecs, document_img_vecs, dim=-1)
            text_score = torch.matmul(query_vecs, document_vecs.T)
            image_score = torch.matmul(query_img_vecs, document_img_vecs.T)

            # score = text_score + image_score
            score = 0.5 * text_score + 0.5 * image_score
            scores.append(score)

        scores = torch.stack(scores)
        # scores = torch.tensor(scores)
        # print(scores)
        # return scores
        return scores
    

    def get_features(self, image, text):
        # negative_images B * N * 3 * 224 * 224
        visual_features = self.BioCLIP.visual(image)
        text_features = self.BioCLIP.text(text)

        return visual_features, text_features
    
    def text_features(self, text):

        text_features = self.BioCLIP.text(text)
        return text_features
    
    def visual_features(self, image):
        
        visual_features = self.BioCLIP.visual(image)
        return visual_features

    def encode_text(self, text):
        text_features = self.BioCLIP.text(text)
        text_features= self.text_linear(text_features)

        return text_features
    

    def encode_image(self, image):
        visual_features = self.BioCLIP.visual(image)
        visual_features= self.image_linear(visual_features)

        return visual_features
    
    

    def cl_loss(self, anchor_image, positive_image, negative_images):
        # negative_images B * N * 3 * 224 * 224
        anchor = self.BioCLIP.visual(anchor_image)
        # anchor =  self.cl_linear(anchor)

        positive = self.BioCLIP.visual(positive_image)
        # positive =  self.cl_linear(positive)

        negatives = []
        for negative_image in negative_images:
            negative = self.BioCLIP.visual(negative_image)
            # negative =  self.cl_linear(negative)
            
            negatives.append(negative)
        
        negatives = torch.stack(negatives)
        loss = self.criterion(anchor, positive, negatives)

        return loss


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
    model = ColBERT.from_pretrained("michiyasunaga/BioLinkBERT-base")

    query = ["Pressure reactivity index or PRx is tool for monitoring patients who have raised intracranial pressure (ICP)",
             "monitoring patients"]
    keys = [
        "caused by pathologies such as a traumatic brain injury or subarachnoid haemorrhage",
    ] * 8 + [
        "in order to guide therapy to protect the brain from damagingly high or low cerebral blood flow."
    ] * 8

    model.to("cuda")

    max_seq_len = 512

    query_outputs = tokenizer(query, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)
    key_outputs = tokenizer(keys, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)

    query_outputs = {k: v.to("cuda") for k, v in query_outputs.items()}
    key_outputs = {k: v.to("cuda") for k, v in key_outputs.items()}

    model(query_outputs, key_outputs)
