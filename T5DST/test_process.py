from config import get_args
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import T5TokenizerFast

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
        for (domain, slot), value in slot_pair.items():
            data.append({
                'domain': domain,
                'slot': slot,
                'dialogue_id': dialog['dialogue_id'],
                'input_text': dialog_history + f' {tokenizer.sep_token} {slot.lower()} of {domain.lower()} : {description[(domain, slot)]}',
            })
    return data

def read_data(split, num):
    data = list()
    for i in tqdm(range(num)):
        dialog = pd.read_json('{}{}/dialogues_{:03d}.json'.format(args.data_dir, split, i + 1), dtype={'dialogue_id': str})
        data.extend(process_dialog(dialog, split))
    return data

seen_set = read_data('test_seen', 16)
seen_pdset = pd.DataFrame(seen_set)
seen_dataset = Dataset.from_dict(seen_pdset)
torch.save(seen_dataset, args.save_dir + 'seen_set_des.pt')

unseen_set = read_data('test_unseen', 5)
unseen_pdset = pd.DataFrame(unseen_set)
unseen_dataset = Dataset.from_dict(unseen_pdset)
torch.save(unseen_dataset, args.save_dir + 'unseen_set_des.pt')