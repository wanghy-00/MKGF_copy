import os
import json
import torch
import numpy as np
from cli import HuatuoChatbot
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from model2 import ColBERT
from transformers import AutoTokenizer, AutoModel
from open_clip import create_model_from_pretrained

bot = HuatuoChatbot("../model/HuatuoGPT-Vision-7B", device='cuda:0')

embedding_model_name = "../model/bge-large-en-v1.5"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

biomed_model, biomed_preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
biomed_model.eval()
biomed_tokenizer = AutoTokenizer.from_pretrained("../model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

biomed_model = ColBERT()
biomed_model.load_state_dict(torch.load("../model/retrieve_top100_epoch_10_lr_0.0001_CL_True.pth", map_location='cuda:0'))

with open("../KG/filtered_knowledge_graph.json", "r", encoding="utf-8") as f:
    knowledge_graph = json.load(f)

with open(f"../dataest/VQA-RAD/rad_rationale.json", "r", encoding='utf-8') as f1:
    data = json.load(f1)

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

def process_question_and_image(question, image_path):
    question_tokens = biomed_tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=256)
    question_image = Image.open(image_path)
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
        
        text_similarity = torch.matmul(biomed_model.encode_text(question_tokens['input_ids']).detach(), entity_text_embedding).item()
        image_similarity = torch.matmul(biomed_model.encode_image(question_image_input).detach(), entity_image_embedding).item()
        
        total_similarity = text_similarity + image_similarity
        
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

for line in tqdm(data):
    question_id = line["qid"]  
    image_file = line["img_name"]
    question_text = line["question"].strip()
    reason = line["reason"].strip()
    answer_type = line["answer_type"]
    gt_value = line['answer'].lower()

    top_indices = retrieve_top_text(reason, knowledge_graph, cached_entity_embeddings)  
    
    question_tokens, question_image_input = process_question_and_image(question_text, os.path.join("../dataest/SLAKE/imgs", image_file))
    
    selected_indices, selected_similarities = rerank_with_multimodal(question_tokens, question_image_input, top_10_indices, knowledge_graph, entity_text_embeddings, entity_image_embeddings)

    retrieved_chunks_with_similarities = [
        {
            "entity": knowledge_graph[idx]['entity'],
            "chunk": knowledge_graph[idx]['chunk'],
            "similarity": similarity
        }
        for idx, similarity in zip(selected_indices, selected_similarities)
    ]

    retrieved_chunks = "\n".join([chunk_info["chunk"] for chunk_info in retrieved_chunks_with_similarities])

    if answer_type == 'OPEN':
        cur_prompt = (
            "You are an expert in medical imaging with extensive knowledge in interpreting imaging results and understanding related medical concepts. Please provide a detailed, thorough, and comprehensive answer to the following open-ended question, incorporating all relevant information. The following retrieved medical knowledge might be helpful: \n"
            + retrieved_chunks +  
            "The above reference information may be useful or it may not be useful.Please provide a thorough, well-reasoned answer, considering all pertinent details and possible implications.\n"
            "Question: " + question_text  
        )


    elif answer_type == 'CLOSED':
        cur_prompt = (
            "You are an expert in medical imaging with extensive knowledge in interpreting imaging results and understanding related medical concepts. Please provide a concise and direct answer to the following closed question, based on the information provided. The following retrieved medical knowledge might be helpful: \n"
            + retrieved_chunks +  
            "The above reference information may be useful or it may not be useful.Please provide a thorough, well-reasoned answer, considering all pertinent details and possible implications.\n"
            "Question: " + question_text  
        )


    image_path = os.path.join("../dataest/SLAKE/imgs", image_file)
    outputs = bot.inference(cur_prompt, image_path)
    
    if isinstance(outputs, list) and len(outputs) > 0:
        pred_value = outputs[0].strip().lower()
    else:
        pred_value = outputs

    if line['answer_type'] == 'OPEN':
        recall_token_num = 0.0
        all_token_num = 0.0
        tokens = split_sentence(line['answer'].lower(), 1).keys()
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
                "question_id": question_id,
                "question": line["question"],
                "image": image_file,
                "answer_type": "OPEN",
                "correct_answer": line['answer'],
                "model_output": outputs,
                "retrieved_chunks": retrieved_chunks_with_similarities,
                "recall": one_recall
            })

    elif answer_type == 'CLOSED':
        closed_total += 1
        if gt_value not in pred_value:
            closed_correct = closed_correct
        else:
            closed_correct += 1
            errors_list.append({
                "question_id": question_id,
                "question": line["question"],
                "image": image_file,
                "answer_type": "CLOSED",
                "correct_answer": line['answer'],
                "retrieved_chunks": retrieved_chunks_with_similarities,
                "model_output": outputs
            })

open_recall = all_recall / all_num
closed_accuracy = closed_correct / closed_total if closed_total > 0 else 0

print(f"Open Recall: {open_recall * 100:.8f}%")
print(f"Closed Accuracy: {closed_accuracy * 100:.8f}%")

with open('huatuo_slake.json', 'w', encoding='utf-8') as f:
    json.dump(errors_list, f, ensure_ascii=False, indent=4)



