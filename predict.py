import sys
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import T5Tokenizer
from transformers import MT5ForConditionalGeneration


input_file = sys.argv[1]
output_file = sys.argv[2]
if input_file:
    eval_train = pd.read_json(path_or_buf=input_file, lines=True)
eval_train = Dataset.from_pandas(eval_train)

checkpoint = 'mt5_5'
tokenizer = T5Tokenizer.from_pretrained(checkpoint)

max_input_length = 256
max_target_length = 64

def preprocess_function_eval(examples):
    model_inputs = tokenizer(
        examples["maintext"], max_length=max_input_length, truncation=True,padding=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=max_target_length, truncation=True,padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MT5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

def chunk(element,batch_size):
    for i in range(0, len(element), batch_size):
        yield element[i: i+batch_size]

titles = []
id = []
def eval_predict(batch_size=16,device=device):

    article_batches = list(chunk(eval_train["maintext"], batch_size))
    title_batches = list(chunk(eval_train["title"], batch_size))
    for article_batches, title_batches in tqdm(zip(article_batches,title_batches), total = len(article_batches)):
        inputs = tokenizer(article_batches, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
        title = model.generate(input_ids=inputs["input_ids"].to(device),attention_mask=inputs["attention_mask"].to(device),length_penalty=2.0
                ,num_beams=4, max_length=150, early_stopping=True,no_repeat_ngram_size=2)
        decoded_title = [tokenizer.decode(s, skip_special_tokens=True,clean_up_tokenization_spaces=True) for s in title]
        decoded_title = [d.replace("<n>"," ") for d in decoded_title]
        titles.append(decoded_title)
    id.append(eval_train["id"])
eval_predict()

title_batch = [i for item in titles for i in item]

submit = pd.DataFrame()
submit['title'] = title_batch
submit['id'] = id[0]
submit['id'] = submit['id'].astype("string")

if output_file:
    submit.to_json(output_file,orient='records', lines=True)
else:
    submit.to_json('predict.jsonl',orient='records', lines=True)
