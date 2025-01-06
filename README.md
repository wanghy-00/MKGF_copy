# MKGF
This is the official github repository for the paper "MKGF:A Multi-modal Knowledge Graph Based RAG Framework to Enhance LVLMs for Medical Visual Question Answering".

We present the source code and release the data.

## Contents

- [MKGF](#MKGF)
  - [Contents](#Contents)
  - [Overview](#Overview)
  - [Dataset](#Dataset)
  - [MMKG](#MMKG)
  - [Method](#Method)

## Overview
![MKFG](https://raw.githubusercontent.com/ehnal/MKGF/main/MKFG.jpg?timestamp=20250101)

We propose a MKGF framework that leverages a multi-modal medicaknowledge graph (MMKG) to relieve the hallucination issue without fine-tuning the abundant parameters of LVLMs. Firstly, we employ a pre-trained text retriever to build question-knowledge relations on training set. Secondly, we train a multi-modal retriever with these relations. Finally, we use it to retrieve question-relevant knowledge and enhance the performance of LVLMs on the test set. To evaluate the effectiveness of MKGF,we conduct extensive experiments on two public datasets Slake and VQA-RAD.

## Dataset
To evaluate the effectiveness of the proposed MKGF framework, we conduct experiments on two public Med-VOA datasets:(1)Slake and (2)VQA-RAD.
For Slake, we use its English version, which contains 642 radiology images and 7,033 question-answer pairs. For VQA-RAD, it contains 315 radiology images and 3,515 question–answer pairs.

## MMKG
We construct the MMKG based on a public medical knowledge graph (MKG), which has 52.6K triples of the head entity, relation and tail entity. 

The textual part of the MMKG is available in /KG.

The pictures in the knowledge graph will be given after the official submission of the paper.

## Method

### Step 0 
### Prepare models,dataest and environment
Download the LVLMs LLava-med-7B or HuatuoGPT-Vision-7B in /model. 

Download the model BGE and BiomedCLIP in /model. for retrieval.

Download the dataest SLAKE in /dataest/SLAKE

Then install the environment in requirements.txt
```
>>> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### Step 1
### Generate rationale
You can run the following script to generate rationale based on SLAKE with GPT-4o.
```
>>> cd rationale
>>> python generate_rationale.py
```
Or you can directly use the file we generated using GPT-4o(/rationale/SLAKE/slake_rationale.json)

### Step 2 
### Train multimodal retriever
You can run the following script to train a multimodal retriever based on BiomedCLIP
```
>>> cd reranker
>>> python -u main.py
```

### Step 3
### Run MKGF
You can run the following script to run and evaluate our method on LLava-med-7B.
```
>>> cd MKGF_llava
>>> python llava_slake.py
```

You can run the following script to run and evaluate our method on HuatuoGPT-Vision-7B.
```
>>> cd MKGF_huatuo
>>> python huatuo_slake.py
```



