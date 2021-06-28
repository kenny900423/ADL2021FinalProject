from config import get_args

import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import T5TokenizerFast
import torch

args = get_args()

tokenizer = T5TokenizerFast.from_pretrained(args.model, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")

schema = pd.read_json(args.schema_path)
schema = schema.set_index('service_name')['slots']


def get_slot(services):
    subset = schema.loc[services].reset_index()
    slot_pair = dict()
    description = dict()
    for i in range(len(subset)):
        domain = subset['service_name'][i]
        for slot in subset['slots'][i]:
            slot_pair[(domain, slot['name'])] = 'none'
            description[(domain, slot['name'])] = slot['description']
    return slot_pair, description

def process_dialog(dialogs, split):
    data = []
    for _, dialog in dialogs.iterrows():
        dialog_history = ''
        slot_pair, description = get_slot(dialog['services'])
        assert len(dialog['turns']) % 2 == 0
        for i in range(0, len(dialog['turns']), 2):
            user, system = dialog['turns'][i], dialog['turns'][i + 1]
            dialog_history += ' user: ' + user['utterance'] + ' system: ' + system['utterance']
            for f in user['frames']:
                domain = f['service']
                if split == 'train' or split == 'dev':
                    for _slot, _value in f['state']['slot_values'].items():
                        assert slot_pair[(domain, _slot)]
                        slot_pair[(domain, _slot)] = _value[0]  # 先取第一個答案
            for (domain, slot), value in slot_pair.items():
                data.append({
                    'input_text': dialog_history + f' {tokenizer.sep_token} {slot.lower()} of {domain.lower()} : {description[(domain, slot)]}',
                    'output_text': f'{value} {tokenizer.eos_token}',
                })
    return data

def read_data(split, num):
    data = list()
    for i in tqdm(range(num)):
        dialog = pd.read_json('{}{}/dialogues_{:03d}.json'.format(args.data_dir, split, i + 1))
        data.extend(process_dialog(dialog, split))
    return data

train_set = read_data('train', 138)
train_pdset = pd.DataFrame(train_set)
train_dataset = Dataset.from_pandas(train_pdset)
torch.save(train_dataset, 'train_set_des.pt')

eval_set = read_data('dev', 20)
eval_pdset = pd.DataFrame(eval_set)
eval_dataset = Dataset.from_pandas(eval_pdset)
torch.save(eval_dataset, 'eval_set_des.pt')