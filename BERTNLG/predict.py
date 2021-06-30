from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import os
import json
import nltk
import random
import sys
from pprint import pprint
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir', default='../../data/data/test_seen/')
parser.add_argument('--model_path', default='./model')
parser.add_argument('--output_path', default='./res.json')
args = parser.parse_args()

chatty = 0.5

nltk.download('punkt')
random.seed(1024)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({
    'sep_token': '<SEP>'
})
tokenizer.pad_token = tokenizer.eos_token

chef = pipeline(
    'text-generation',
    model=args.model_path,
    tokenizer=tokenizer,
    config={
        'max_length': 512,
    },
    device=0
)

test_dir = args.data_dir

nlg = {}

for file in tqdm(os.listdir(test_dir), file=sys.stdout):
    with open(test_dir + file, 'r') as f:
        dialogues = json.load(f)
    
    for dialogue in tqdm(dialogues, file=sys.stdout):
        dialogue_id = dialogue['dialogue_id']
        
        dialogue_res = {}

        for turn in dialogue['turns']:
            turn_id = turn['turn_id']

            if turn['speaker'] != 'SYSTEM':
                user_uttr = turn['utterance']
                continue
            
            sys_uttr = turn['utterance']

            if random.random() >= chatty:
                text_in = user_uttr + '<SEP>' + sys_uttr + '<SEP>'

                begin = chef(
                    text_in,
                    return_full_text=False,
                    max_length=100,
                    num_beams=3, 
                    no_repeat_ngram_size=2, 
                    early_stopping=True,
                    length_penalty=0.5
                )[0]['generated_text']

                begin = nltk.sent_tokenize(begin)[0].strip().capitalize()
            else:
                begin = ''

            if random.random() >= chatty:
                text_in = user_uttr + '<SEP>' + sys_uttr + '<SEP>' + begin + '<SEP>'

                end = chef(
                    text_in,
                    return_full_text=False,
                    max_length=130,
                    num_beams=3, 
                    no_repeat_ngram_size=2, 
                    early_stopping=True,
                    length_penalty=0.5
                )[0]['generated_text']

                end = nltk.sent_tokenize(end)[0].strip().capitalize()
            else:
                end = ''

            dialogue_res[turn_id] = {
                'begin': begin,
                'end': end,
                'mod': ''
            }
        
        nlg[dialogue_id] = dialogue_res

with open(args.output_path, 'w') as f:
    json.dump(nlg, f)