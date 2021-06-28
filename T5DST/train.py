from config import get_args
import torch
import pytorch_lightning as pl
from transformers import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from functools import partial
from transformers import T5TokenizerFast, T5ForConditionalGeneration

args = get_args()

train_dataset = torch.load('train_set_des.pt')
eval_dataset = torch.load('eval_set_des.pt')

def collate_fn(data, tokenizer):
    batch_data, unzip_data = {}, {}
    for key in data[0].keys():
        unzip_data[key] = [d[key] for d in data]

    input_batch = tokenizer(unzip_data["input_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(unzip_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data

class DST_Seq2Seq(pl.LightningModule):

    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args.lr


    def training_step(self, batch, batch_idx):
        self.model.train()
        outputs = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"])
        loss = outputs.loss
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        outputs = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"])
        loss = outputs.loss
        return {'val_loss': loss, 'log': {'val_loss': loss}}

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)


seed_everything(args.seed)
tokenizer = T5TokenizerFast.from_pretrained('t5-base', bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.resize_token_embeddings(new_num_tokens=len(tokenizer))

task = DST_Seq2Seq(args, tokenizer, model)

train_dataloader = DataLoader(train_dataset, args.train_batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=args.num_workers)
eval_dataloader = DataLoader(eval_dataset, args.eval_batch_size, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=args.num_workers)

model_path = args.save_dir + 'myt5'
trainer = Trainer(
            default_root_dir=model_path,
            max_epochs=args.n_epochs, 
            gpus=args.n_gpus,
            accelerator="dp"
        )

trainer.fit(task, train_dataloader, eval_dataloader)
task.model.save_pretrained(model_path)
task.tokenizer.save_pretrained(model_path)