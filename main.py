from transformers import AutoTokenizer, AutoModelForSequenceClassification
from entity_relationship_model import EntityRelationshipDataset, EntityRelationshipModel, train_model
from data_preprocessing import get_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd




# Define constants
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_CLASSES = 20

tokenizer = AutoTokenizer.from_pretrained("./roberta-chinese-finetuned")

data_path = "./data/entity_relation_data.jsonl"
data, entity_labels, relation_labels = get_dataset(data_path)
labels = [relation_labels.index(obs["relation"][0]["type"]) if obs["relation"] else 0 for obs in data]
train_texts, val_texts, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

train_dataset = EntityRelationshipDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = EntityRelationshipDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EntityRelationshipModel(tokenizer, NUM_CLASSES).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_model(model, train_loader, val_loader, optimizer, criterion)