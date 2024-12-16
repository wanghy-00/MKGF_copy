import os
from cli import HuatuoChatbot
import json
import torch
import pandas as pd
import numpy as np
from open_clip import create_model_from_pretrained
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from io import BytesIO
from model import ColBERT

bot = HuatuoChatbot("../model/HuatuoGPT-Vision-7B", device='cuda:0')

embedding_model_name = "../model/bge-large-en-v1.5"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

biomed_model, biomed_preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
biomed_model.eval()
biomed_tokenizer = AutoTokenizer.from_pretrained("../model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

biomed_model = ColBERT()
biomed_model.load_state_dict(torch.load("../model/retrieve_top100_epoch_10_lr_0.0001_CL_True.pth", map_location='cuda:0'))

splits = {'train': '../dataest/VQA-RAD/data/train-00000-of-00001-eb8844602202be60.parquet', 'test': '../dataest/VQA-RAD/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet'}
questions = pd.read_parquet("../dataest/VQA-RAD/" + splits["test"])

with open("../KG/filtered_knowledge_graph.json", "r", encoding="utf-8") as f:
    knowledge_graph = json.load(f)

with open('../dataest/VQA-RAD/rad_rationale.json', 'r', encoding='utf-8') as f:
    rationale_data = json.load(f)

def compute_text_embedding(text):
    tokens = embedding_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    embedding = embedding_model(**tokens).last_hidden_state.mean(dim=1).detach()  
    return embedding

def compute_text_embedding(text):
    inputs = embedding_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def split_sentence(sentence, n):
    words = defaultdict(int)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=-1)

def preprocess_knowledge_graph(knowledge_graph):
    cached_entity_embeddings = {}
    
    for entity in knowledge_graph:
        entity_chunk = entity["chunk"]
        entity_text_embedding = compute_text_embedding(entity_chunk)  
        
        if isinstance(entity_text_embedding, np.ndarray):
            entity_text_embedding = torch.tensor(entity_text_embedding)
        
        cached_entity_embeddings[entity["entity"]] = entity_text_embedding  

    return cached_entity_embeddings

def retrieve_top_text(question, knowledge_graph, cached_entity_embeddings):
    question_embedding = compute_text_embedding(question)
    
    if isinstance(question_embedding, np.ndarray):
        question_embedding = torch.tensor(question_embedding)

    text_similarities = []

    for entity in knowledge_graph:
        entity_text_embedding = cached_entity_embeddings[entity["entity"]]

        if isinstance(entity_text_embedding, np.ndarray):
            entity_text_embedding = torch.tensor(entity_text_embedding)

        similarity = torch.nn.functional.cosine_similarity(question_embedding, entity_text_embedding, dim=-1).item()
        text_similarities.append(similarity)
    
    top_indices = sorted(range(len(text_similarities)), key=lambda i: text_similarities[i], reverse=True)[:8]
    return top_indices

def process_question_and_image(question, image):
    question_tokens = biomed_tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=256)
    question_image = image
    question_image_input = biomed_preprocess(question_image).unsqueeze(0)
    return question_tokens, question_image_input

def process_knowledge_graph(knowledge_graph):
    biomed_model.eval()

    entity_text_embeddings = []
    entity_image_embeddings = []
    
    target_dim = 512  

    with torch.no_grad():
        for entity in knowledge_graph:
            entity_chunk = entity["chunk"]
            
            entity_text_tokens = biomed_tokenizer(entity_chunk, return_tensors='pt', padding=True, truncation=True, max_length=256)
            entity_text_embedding = biomed_model.encode_text(entity_text_tokens['input_ids']).detach()

            if entity_text_embedding.shape[1] != target_dim:
                if entity_text_embedding.shape[1] < target_dim:
                    padding = torch.zeros((entity_text_embedding.shape[0], target_dim - entity_text_embedding.shape[1]))
                    entity_text_embedding = torch.cat([entity_text_embedding, padding], dim=1)
                else:
                    entity_text_embedding = entity_text_embedding[:, :target_dim]  

            if "image" in entity:
                entity_image_path = os.path.join("../entity_img", entity["image"])
                
                if os.path.exists(entity_image_path):
                    entity_image = Image.open(entity_image_path)
                    entity_image_input = biomed_preprocess(entity_image).unsqueeze(0)
                    entity_image_embedding = biomed_model.encode_image(entity_image_input).detach()

                    if entity_image_embedding.shape[1] != target_dim:
                        if entity_image_embedding.shape[1] < target_dim:
                            padding = torch.zeros((entity_image_embedding.shape[0], target_dim - entity_image_embedding.shape[1]))
                            entity_image_embedding = torch.cat([entity_image_embedding, padding], dim=1)
                        else:
                            entity_image_embedding = entity_image_embedding[:, :target_dim]  

                    entity_image_embeddings.append(entity_image_embedding)
                else:
                    print(f"Warning: Image file not found: {entity_image_path}")
                    entity_image_embeddings.append(torch.zeros((1, target_dim)))  
            else:
                entity_image_embeddings.append(torch.zeros((1, target_dim)))  

            entity_text_embeddings.append(entity_text_embedding)
    
    return torch.cat(entity_text_embeddings), torch.cat(entity_image_embeddings)

def rerank_with_multimodal(question_tokens, question_image_input, top_indices, knowledge_graph, entity_text_embeddings, entity_image_embeddings):
    text_similarities = []
    image_similarities = []
    total_similarities = []
    
    for idx in top_indices:
        entity_text_embedding = entity_text_embeddings[idx]
        entity_image_embedding = entity_image_embeddings[idx]
        
        text_similarity = cosine_similarity(biomed_model.encode_text(question_tokens['input_ids']).detach(), entity_text_embedding).item()
        image_similarity = cosine_similarity(biomed_model.encode_image(question_image_input).detach(), entity_image_embedding).item()
        
        total_similarity = text_similarity +  image_similarity
        
        text_similarities.append(text_similarity)
        image_similarities.append(image_similarity)
        total_similarities.append(total_similarity)
    
    top_3_indices = sorted(range(len(total_similarities)), key=lambda i: total_similarities[i], reverse=True)[:3]
    
    final_indices = [top_indices[i] for i in top_3_indices]
    
    return final_indices, [total_similarities[i] for i in top_3_indices]

all_num = 0
all_recall = 0.0
closed_correct = 0
closed_total = 0

cached_entity_embeddings = preprocess_knowledge_graph(knowledge_graph)
entity_text_embeddings, entity_image_embeddings = process_knowledge_graph(knowledge_graph)

errors_list = []

for qid in tqdm(range(len(questions))):

    question = questions['question'][qid]
    ans = questions['answer'][qid]
    rationale = rationale_data[qid]['rationale']
    image = Image.open(BytesIO(questions['image'][qid]['bytes']))
    
    top_indices = retrieve_top_text(rationale, knowledge_graph, cached_entity_embeddings)  
    
    question_tokens, question_image_input = process_question_and_image(rationale, image)
    
    selected_indices, selected_similarities = rerank_with_multimodal(question_tokens, question_image_input, top_8_indices, knowledge_graph, entity_text_embeddings, entity_image_embeddings)

    retrieved_chunks_with_similarities = [
        {
            "entity": knowledge_graph[idx]['entity'],
            "chunk": knowledge_graph[idx]['chunk'],
            "similarity": similarity
        }
        for idx, similarity in zip(selected_indices, selected_similarities)
    ]

    retrieved_chunks = "\n".join([chunk_info["chunk"] for chunk_info in retrieved_chunks_with_similarities])


    if ans.lower() == "yes" or ans.lower() == "no":
        if all(similarity < 1.2 for similarity in selected_similarities):
            cur_prompt =  question
        else:
            cur_prompt = (
                "You are an expert in medical imaging with extensive knowledge in interpreting imaging results and understanding related medical concepts. The following retrieved medical knowledge might be helpful: \n"
                + retrieved_chunks +  
                "Please return yes or no.\n"
                "Question: " + question
            )

    else:
        cur_prompt = (
            "You are an expert in medical imaging with extensive knowledge in interpreting imaging results and understanding related medical concepts. The following retrieved medical knowledge might be helpful: \n"
            + retrieved_chunks +  
            "\nPlease provide a detailed ,well-reasoned and comprehensive answer to the following open-ended question.\n"
            "Question: " + question 
        )
    
    outputs = bot.inference(cur_prompt, [image])

    if isinstance(outputs, list) and len(outputs) > 0:
        pred_value = outputs[0].strip().lower()
    else:
        pred_value = outputs

    if ans.lower() == "yes" or ans.lower() == "no":
        closed_total += 1
        if ans.lower() not in pred_value:
            errors_list.append({
                "question": question,
                "answer_type": "CLOSED",
                "retrieved_chunks": retrieved_chunks_with_similarities,
                "correct_answer": ans.lower(),
                "model_output": outputs
            })
        else:
            closed_correct += 1

    else:
        recall_token_num = 0.0
        all_token_num = 0.0
        tokens = split_sentence(ans.lower(), 1).keys()
        for token in tokens:
            all_token_num += 1
            if token[-1] == ',':
                token = token[:-1]
            if token in pred_value:
                recall_token_num += 1
    
        one_recall = recall_token_num / all_token_num
        
        all_num += 1
        all_recall += one_recall

        if one_recall < 1.0:
            errors_list.append({
                "question": question,
                "answer_type": "OPEN",
                "retrieved_chunks": retrieved_chunks_with_similarities,
                "correct_answer": ans.lower(),
                "model_output": outputs,
                "recall": one_recall
            })

open_recall = all_recall / all_num
closed_accuracy = closed_correct / closed_total if closed_total > 0 else 0

print(f"Open Recall: {open_recall * 100:.8f}%")
print(f"Closed Accuracy: {closed_accuracy * 100:.8f}%")

with open('huatuo_rad.json', 'w', encoding='utf-8') as f:
    json.dump(errors_list, f, ensure_ascii=False, indent=4)



