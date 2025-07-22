from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch

from data_preprocessing import preprocess_text, chunk_text, TextClassificationDataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import pandas as pd

# Constants
MAX_LEN = 128
BATCH_SIZE = 16

# Initialize RoBERTa Tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base-chinese")
text_data = pd.read_csv('train_data.csv')
text_data = text_data.sample(frac=0.3)
dataset = Dataset.from_dict({'text': text_data['text'].tolist(), 'label': text_data['pretrained_label'].tolist()})
chunked_dataset = dataset.map(chunk_text, batched=True, remove_columns=['text'])

df = chunked_dataset.to_pandas()
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['pretrained_label'].tolist(), test_size=0.2, random_state=42
)
train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=MAX_LEN)
val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=MAX_LEN)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'label': train_labels
})
val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'label': val_labels
})

model = RobertaForSequenceClassification.from_pretrained("roberta-base-chinese", num_labels=2)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save the fine-tuned model and tokenizer
tokenizer.save_pretrained("./roberta-chinese-finetuned")
model.save_pretrained("./roberta-chinese-finetuned")
