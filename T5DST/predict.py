from config import get_args
import torch
from tqdm import tqdm
from functools import partial
from transformers import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from transformers import T5TokenizerFast, T5ForConditionalGeneration

args = get_args()

tokenizer = T5TokenizerFast.from_pretrained(args.save_dir + 'model')
model = T5ForConditionalGeneration.from_pretrained(args.save_dir + 'model')

seen_dataset = torch.load(args.save_dir + 'seen_set_des.pt')
unseen_dataset = torch.load(args.save_dir + 'unseen_set_des.pt')

def collate_fn(data, tokenizer):
    batch_data, unzip_data = {}, {}
    for key in data[0].keys():
        unzip_data[key] = [d[key] for d in data]

    input_batch = tokenizer(unzip_data["input_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    input_batch["dialogue_id"] = unzip_data["dialogue_id"]
    input_batch["domain"] = unzip_data["domain"]
    input_batch["slot"] = unzip_data["slot"]

    return input_batch

seen_dataloader = DataLoader(seen_dataset, args.test_batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=args.num_workers)
unseen_dataloader = DataLoader(unseen_dataset, args.test_batch_size, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=args.num_workers)

def gen_ans(dataloader):
    model.cuda()
    model.eval()
    ans = {}
    for i in tqdm(dataloader):
        output = model.generate(i['input_ids'].cuda(), num_beams=args.num_beams, length_penalty=0.5, repetition_penalty=1.2)
        i['out_text'] = tokenizer.batch_decode(output, skip_special_tokens=True)
        for dialogue_id, domain, slot, out_text in zip(i['dialogue_id'], i['domain'], i['slot'], i['out_text']):
            ans[dialogue_id] = ans.get(dialogue_id, {})
            if 'none' in out_text:
                continue
            ans[dialogue_id][f'{domain}-{slot}'] = out_text
    return ans

def write_csv(ans, output_path):
    ans = sorted(ans.items(), key=lambda x: x[0])
    with open(output_path, 'w') as f:
        f.write('id,state\n')
        for dialogue_id, states in ans:
            if len(states) == 0:  # no state ?
                str_state = 'None'
            else:
                states = sorted(states.items(), key=lambda x: x[0])
                str_state = ''
                for slot, value in states:
                    # NOTE: slot = "{}-{}".format(service_name, slot_name)
                    str_state += "{}={}|".format(
                            slot.lower(), value.replace(',', '_').lower())
                str_state = str_state[:-1]
            f.write('{},{}\n'.format(dialogue_id, str_state))

seen_ans = gen_ans(seen_dataloader)
write_csv(seen_ans, args.save_dir + 'seen_ans.csv')

unseen_ans = gen_ans(unseen_dataloader)
write_csv(unseen_ans, args.save_dir + 'unseen_ans.csv')