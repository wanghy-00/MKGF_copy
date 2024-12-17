import json
import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

MAX_TOKENS = 1024

KEY = ""
BASE_URL = ""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def callAPI_meta(messages, key=KEY, base_url=BASE_URL, max_tokens=MAX_TOKENS):
    client = OpenAI(api_key=key, base_url=base_url)
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        temperature=0,
        seed=42,
        max_tokens=max_tokens,
        )
    return response.choices[0].message.content

if __name__ == '__main__':
    image_path = '../dataest/SLAKE/imgs'

    with open(f"../dataest/SLAKE/test.json","r",encoding='utf-8') as f1:
        questions = json.load(f1)
    
    records = []
    
    for line in tqdm(questions):
        record = dict()

        image_file = line["img_name"]
        qs = line["question"].strip()

        base64_image = encode_image(os.path.join(image_path, image_file))
        Instruction = '''The following is a question related to medical imaging. Please generate a description of the image based on the question. Only output the description.'''
        Prompt = Instruction + '\n' + "Question: " + qs  

        messages = [
            {
                "role": "user",
                "content": [ 
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},}, 
                    {"type": "text", "text": Prompt},
                    ],
            }
        ]

        rationale = callAPI_meta(messages)

        record['image'] = os.path.join(image_path, image_file)
        record['question'] = qs
        record['rationale'] = rationale
        records.append(record)

    with open(f"slake_rationale.json", "w", encoding='utf-8') as f1:
        json.dump(records, f1)