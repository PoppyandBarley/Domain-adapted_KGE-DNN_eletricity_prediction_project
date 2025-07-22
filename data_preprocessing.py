import numpy as np
import pandas as pd
import re
import json
from typing import List, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Constants
MAX_LEN = 128
# ------------------------------------------
# Text Preprocessing Function
# ------------------------------------------
def preprocess_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)                        
    text = re.sub(r'http[s]?://\S+', '', text)               
    text = re.sub(r'\$.*?\$', '', text)                      
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)              
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)               
    text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fff\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str) -> List[List[int]]:
    est_max_chars = MAX_LEN * 4
    return [text[i:i + est_max_chars] for i in range(0, len(text), est_max_chars)]

def get_dataset(path: str) -> Tuple[List[dict], List[str], List[str]]:
    data = []
    ne_set = set()
    rel_set = set()
    n_obs = 0
    n_broken = 0

    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc=f"Loading {path}"):
            skip_obs = False
            obs = json.loads(line)
            sent_text = obs.get("text", "")
            entity_mentions = obs.get("entity", [])
            relation_mentions = obs.get("relation", [])

            for ne in entity_mentions:
                ne_set.add(ne["label"])
                if ne["text"] not in sent_text:
                    n_broken += 1
                    skip_obs = True

            for rel in relation_mentions:
                rel_set.add(rel["type"])
                if rel["from"] not in sent_text or rel["to"] not in sent_text:
                    n_broken += 1
                    skip_obs = True

            if skip_obs:
                continue

            n_obs += 1
            data.append(obs)

    print(f"++++ Valid observations: {n_obs}")
    print(f"++++ Skipped (broken) observations: {n_broken}")

    return data, sorted(list(ne_set)), sorted(list(rel_set))

