import sys
import json
import nltk
import torch
import argparse
import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_metric
from transformers import DataCollatorForSeq2Seq
from transformers import MT5ForConditionalGeneration
from transformers import TFMT5EncoderModel, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

train_file = sys.argv[1]
eval_file = sys.argv[2]

if train_file:
    with jsonlines.open(train_file) as f:
        data_train = [obj for obj in f]
if eval_file:
    with jsonlines.open(eval_file) as f:
        data_eval = [obj for obj in f]

train = pd.DataFrame(data_train)
eval = pd.DataFrame(data_eval)
train.drop('date_publish',inplace=True,axis=1)
train.drop('split',inplace=True,axis=1)
train.drop('source_domain',inplace=True,axis=1)

eval.drop('date_publish',inplace=True,axis=1)
eval.drop('split',inplace=True,axis=1)
eval.drop('source_domain',inplace=True,axis=1)

train = Dataset.from_pandas(train)
eval = Dataset.from_pandas(eval)

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

max_input_length = 256
max_target_length = 64

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["maintext"], max_length=max_input_length, truncation=True,padding=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=max_target_length, truncation=True,padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train.map(preprocess_function, batched=True,remove_columns=train.column_names)
tokenized_eval = eval.map(preprocess_function, batched=True,remove_columns=eval.column_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
model = model.to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

rouge_score = load_metric("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=3,
    num_train_epochs=25,
    adam_beta1=0.9,
    fp16=True,
    gradient_accumulation_steps=1,
    eval_accumulation_steps = 1,
    predict_with_generate = True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics = compute_metrics,
)

trainer.train()
model_dir = 'mt5_5'
trainer.save_model(model_dir)
