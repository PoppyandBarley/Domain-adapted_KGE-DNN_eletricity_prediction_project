import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import numpy as np

class EntityRelationshipDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_len: int, 
                 entity_types: List[str], relation_types: List[str]):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.entity2id = {e: i for i, e in enumerate(entity_types)}
        self.id2entity = {i: e for i, e in enumerate(entity_types)}
        self.relation2id = {r: i for i, r in enumerate(relation_types)}
        self.id2relation = {i: r for i, r in enumerate(relation_types)}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        entities = item["entities"]
        relations = item.get("relations", [])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        offset_mapping = encoding["offset_mapping"].squeeze().tolist()
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        entity_labels = [0] * len(offset_mapping)
        
        token_entities = []
        
        for entity in entities:
            start_char = entity["start_idx"]
            end_char = entity["end_idx"]
            label = entity["label"]
            
            start_token, end_token = None, None
            for i, (start, end) in enumerate(offset_mapping):
                if start <= start_char < end and start_token is None:
                    start_token = i
                if start < end_char <= end:
                    end_token = i
                if start_token is not None and end_token is not None:
                    break
            
            if start_token is not None and end_token is not None:
                entity_labels[start_token] = self.entity2id[label] * 2 + 1
                for i in range(start_token + 1, end_token + 1):
                    entity_labels[i] = self.entity2id[label] * 2 + 2
                
                token_entities.append({
                    "start_token": start_token,
                    "end_token": end_token,
                    "label": label,
                    "text": text[start_char:end_char]
                })
        
        entity_labels = entity_labels[:self.max_len]
        entity_labels += [0] * (self.max_len - len(entity_labels))
        entity_labels = torch.tensor(entity_labels)
        
        relation_matrix = np.zeros((self.max_len, self.max_len), dtype=np.long)
        
        for relation in relations:
            from_entity = relation["from"]
            to_entity = relation["to"]
            relation_type = relation["type"]
            
            from_entity_info = next((e for e in token_entities if e["text"] == from_entity), None)
            to_entity_info = next((e for e in token_entities if e["text"] == to_entity), None)
            
            if from_entity_info and to_entity_info:
                from_token = from_entity_info["start_token"]
                to_token = to_entity_info["start_token"]
                relation_matrix[from_token, to_token] = self.relation2id[relation_type]
        
        relation_matrix = torch.tensor(relation_matrix)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_labels": entity_labels,
            "relation_matrix": relation_matrix,
            "text": text,
            "original_entities": entities,
            "original_relations": relations,
            "offset_mapping": offset_mapping
        }

class BiAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(BiAttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, head_output, tail_output):
        head_att = self.attention(head_output)  
        tail_att = self.attention(tail_output)  
        
        interaction = torch.bmm(head_att, tail_att.transpose(1, 2)) 
        interaction_softmax = torch.softmax(interaction, dim=-1)
        return interaction_softmax

class EntityRelationshipModel(nn.Module):
    def __init__(self, roberta_model, hidden_size: int, 
                 num_entity_types: int, num_relation_types: int):
        super(EntityRelationshipModel, self).__init__()
        self.roberta = roberta_model
        self.num_entity_types = num_entity_types
        self.num_relation_types = num_relation_types
        
        self.entity_classifier = nn.Sequential(
            nn.Linear(roberta_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_entity_types * 2 + 1)  # BIO labels
        )
        
        self.head_mlp = nn.Sequential(
            nn.Linear(roberta_model.config.hidden_size, hidden_size),
            nn.ReLU()
        )
        self.tail_mlp = nn.Sequential(
            nn.Linear(roberta_model.config.hidden_size, hidden_size),
            nn.ReLU()
        )
        self.bi_attention = BiAttentionLayer(hidden_size)
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # head, tail and attention features
            nn.ReLU(),
            nn.Linear(hidden_size, num_relation_types)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        entity_logits = self.entity_classifier(sequence_output)  # [batch, seq_len, num_entity_types*2+1]
        
        head_output = self.head_mlp(sequence_output)  # [batch, seq_len, hidden_size]
        tail_output = self.tail_mlp(sequence_output)  # [batch, seq_len, hidden_size]
        
        attention_output = self.bi_attention(head_output, tail_output)  # [batch, seq_len, seq_len]
        
        batch_size, seq_len, hidden_size = sequence_output.size()
        
        head_expanded = head_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq_len, seq_len, hidden_size]
        tail_expanded = tail_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch, seq_len, seq_len, hidden_size]
        relation_features = torch.cat([
            head_expanded,
            tail_expanded,
            attention_output.unsqueeze(-1).expand(-1, -1, -1, hidden_size)
        ], dim=-1)
        
        relation_logits = self.relation_classifier(relation_features)  # [batch, seq_len, seq_len, num_relation_types]
        
        return {
            "entity_logits": entity_logits,
            "relation_logits": relation_logits
        }

def train_model(model, train_loader, val_loader, optimizer, criterion, save_path="replaced_with_your_path.pt"):
    NUM_EPOCHS = 100
    NUM_CLASSES = 20
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))
            entity_loss = criterion(logits["entity_logits"], batch["entity_labels"])
            relation_loss = criterion(logits["relation_logits"], batch["relation_matrix"])
            
            total_loss = entity_loss + relation_loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

def load_model(model_class, roberta_model, hidden_size, num_classes, your_path):
    model = model_class(roberta_model, hidden_size, num_classes)
    model.load_state_dict(torch.load(your_path))
    model.eval()
    print(f"Loaded model from {your_path}")
    return model

def predict(model, tokenizer, texts, max_len=128):
    model.eval()
    predictions = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            predictions.append(pred)
    return predictions

if __name__ == "__main__":
    your_classes = 20
    tokenizer = AutoTokenizer.from_pretrained("./roberta-chinese-finetuned")
    model = EntityRelationshipModel(tokenizer, hidden_size=128, num_classes=your_classes)
    model = load_model(model, "replaced_with_your_path.pt")
    sample_texts = ["广西电力交易进行中，南方电网很棒", "广西政府发布广告，对电力交易平台进行优化"]
    predictions = predict(model, tokenizer, sample_texts)
    print("Predictions:", predictions)
