import os
import json
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load, load_metric
from transformers import AutoTokenizer, AutoModelWithLMHead, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir', default='../../data/data/train')
args = parser.parse_args()

train_dir = args.data_dir
n_begin = 0
n_end = 0

data = {
    'id': [],
    'user_utterance': [],
    'system_utterance': [],
    'begin': [],
    'end': []
}

sys_uttr = user_uttr = ""

for file in tqdm(os.listdir(train_dir)):
    with open(train_dir + file, 'r') as f:
        dialogues = json.load(f)
    
    for dialogue in dialogues:
        for turn in dialogue['turns']:
            if turn['speaker'] != 'SYSTEM':
                user_uttr = turn['utterance']
                continue

            sys_uttr = turn['utterance']
            begin_chit = end_chit = ""

            for begin in turn.get('beginning', []):
                if begin['label'] == 'good':
                    begin_chit = begin['candidate']
                    break
                
            for end in turn.get('end', []):
                if end['label'] == 'good':
                    end_chit = end['candidate']
                    break
            
            if begin_chit != "" or end_chit != "":
                data['id'].append(len(data['id']))
                data['user_utterance'].append(user_uttr)
                data['system_utterance'].append(sys_uttr)
                data['begin'].append(begin_chit)
                data['end'].append(end_chit)

dataset = Dataset.from_dict(data)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({
    'sep_token': '<SEP>'
})
tokenizer.pad_token = tokenizer.eos_token

max_length = 1024

def preprocess(examples):
    inputs = list(map(
        lambda x: (x[0] + '<SEP>' + x[1] + '<SEP>' + x[2] + '<SEP>' + x[3]),
        zip(
            examples['user_utterance'],
            examples['system_utterance'],
            examples['begin'],
            examples['end']
        )
    ))
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

    return model_inputs

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelWithLMHead.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir='./gpt2_nlg',
    overwrite_output_dir=True,
    evaluation_strategy='steps',
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_steps=4000,
    save_steps=8000,
    warmup_steps=5000,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

trainer.train()
trainer.save_model('model')