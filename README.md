# MKGF
This is the official github repository for the paper "MKGF:AMulti-modal Knowledge Graph Based RAG Framework to Enhance LVLMs for Medical Visual Question Answering".

We present the source code and release the data.

## Contents

- [MKGF](#MKGF)
  - [Contents](#Contents)
  - [Overview](#Overview)
  - [Dataset](#Dataset)
  - [MMKG](#MMKG)
  - [Method](#Method)

## Overview
<img src="blob:https://github.com/1cf904b3-1f82-4da6-9409-99531b76b564"/>

Medical visual question answering(MedVQA) is a challenging task that requires models to understand medical images and return accurate responses for the given questions. Most recent methods focus on transferring general-domain large vision-language models (LVLMs) to the medical domain by constructing medical instruction datasets and in-context learning. However,the performance of these methods are limited due to the hallucination issue of LVLMs. In addition.fine-tuning the abundant parameters of LVLMs on medical instruction datasets is high time andeconomic cost. Hence, we propose a MKGF framework that leverages a multi-modal medicaknowledge graph (MMKG) to relieve the hallucination issue without fine-tuning the abundant parameters of LVLMs. Firstly, we employ a pre-trained text retriever to build question-knowledge relations on training set. Secondly, we train a multi-modal retriever with these relations. Finally, we use it to retrieve question-relevant knowledge and enhance the performance of LVLMs on the test set. To evaluate the effectiveness of MKGF,we conduct extensive experiments on two public datasets Slake and VQA-RAD.

## Dataset
To evaluate the effectiveness of the proposed MKGF framework, we conduct experiments on two public Med-VOA datasets:(1)Slake and (2)VQA-RAD.For Slake, we use its English version, which contains 642 radiology images and 7,033 question-answer pairs. For VQA-RAD, it contains 315 radiology images and 3,515 question–answer pairs.

## MMKG
We construct the MMKG based on a public medical knowledge graph (MKG), which has 52.6K triples of the head entity, relation and tail entity. 

The textual part of the MMKG is available in /KG.

The image part of the MMKG is available in /entity_img.

## Method
Download the LVLMs LLava-med-7B or HuatuoGPT-Vision-7B in /model. 

Download the model BGE and BiomedCLIP in /model. for retrieval.

Then install the environment in requirements.txt
```
>>> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

You can run the following script to run and evaluate our method on LLava-med-7B.
```
>>> cd LLaVA-Med
>>> python llava_slake.py
>>> python llava_rad.py
```

You can run the following script to run and evaluate our method on HuatuoGPT-Vision-7B.
```
>>> cd HuatuoGPT-Vision
>>> python huatuo_slake.py
>>> python huatuo_rad.py
```


