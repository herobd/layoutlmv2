# -*- coding: utf-8 -*-
import numpy as np
import editdistance
import sys
from PIL import Image
experiment = sys.argv[1]

model_checkpoint = "../pairing/cache_huggingface/layoutlmv2-base-uncased"
batch_size = 5 #if experiment == 'load' else 3

"""## Analysis

Let's load the DocVQA validation split. You can download the DocVQA data (after registration) [here](https://rrc.cvc.uab.es/?ch=17).
"""

from docvqa_dataset import DocVQADataset, collate
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = DocVQADataset('train',load_ocr=experiment=='load')

#print('TEST NUM WORKERS 0')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=9,collate_fn=collate, shuffle=True)
batch = next(iter(dataloader))

import json
with open('../data/DocVQA/val/val_v1.0.json') as f:
  valid_data = json.load(f)
import pandas as pd
valid_df = pd.DataFrame(valid_data['data'])
valid_df.head()
root_dir = '../data/DocVQA/train/'
valid_root_dir = '../data/DocVQA/val/'
from datasets import Dataset
valid_dataset = Dataset.from_pandas(valid_df.iloc[:])
def get_image(examples):
    
  images = [valid_root_dir + image_file for image_file in examples['image']]
  examples['image']=images
  return examples

valid_images = valid_dataset.map(get_image, batched=True, batch_size=2)


#for k,v in batch.items():
"""## Inference

After training, you can perform inference as follows:

1. Take an image + question, prepare it for the model using `LayoutLMv2Processor`. The processor will apply the feature extractor and tokenizer in a sequence, to get all required inputs you need.
2. Forward it through the model.
3. The model returns `start_logits` and `end_logits`, which indicate which token is at the start of the answer and which token is at the end of the answer. Both have shape (batch_size, sequence_length).
4. You can take an argmax on the last dimension of both the `start_logits` and `end_logits` to get the predicted `start_idx` and `end_idx`.
5. You can then decode the answer as follows: `processor.tokenizer.decode(input_ids[start_idx:end_idx+1])`.


"""

## step 1: pick a random example
#example = data['data'][10]
#root_dir = '/content/drive/MyDrive/LayoutLMv2/Tutorial notebooks/DocVQA/val/'
#question = example['question']
#image = Image.open(root_dir + example['image']).convert("RGB")
#print(question)
#image

from transformers import LayoutLMv2Processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

def run(image,question):
    if isinstance(image,str):
        image = Image.open(image).convert("RGB")
    # prepare for the model
    encoding = processor(image, question, return_tensors="pt",max_length=512,truncation=True)
    #print(encoding.keys())

    """Note that you can also verify what the processor has created, by decoding the `input_ids` back to text:"""

    #print(processor.tokenizer.decode(encoding.input_ids.squeeze()))

    # step 2: forward pass

    for k,v in encoding.items():
      encoding[k] = v.to(model.device)

    outputs = model(**encoding)

    # step 3: get start_logits and end_logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # step 4: get largest logit for both
    predicted_start_idx = start_logits.argmax(-1).item()
    predicted_end_idx = end_logits.argmax(-1).item()
    #print("Predicted start idx:", predicted_start_idx)
    #print("Predicted end idx:", predicted_end_idx)

    # step 5: decode the predicted answer
    return processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx:predicted_end_idx+1])
#  print(k, v.shape)

#idx = 2

#tokenizer.decode(batch['input_ids'][2])

#start_position = batch['start_positions'][idx].item()
#end_position = batch['end_positions'][idx].item()

#tokenizer.decode(batch['input_ids'][idx][start_position:end_position+1])

"""## Train a model

Next, we can fine-tune the model on our dataset.
"""

from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
#optimizer = AdamW(model.parameters(), lr=5e-6)

try:
    loading = torch.load('state_latest{}.pth'.format(experiment))
    start_epoch = loading['epoch']
    start_idx = loading['idx']

    optimizer.load_state_dict(loading['optimizer'])
    model.load_state_dict(loading['state_dict'])
except FileNotFoundError:
    start_epoch=0
    start_idx=-1


#with torch.cuda.device(experiment):
if experiment=='load':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, dict):
            for name,subparam in param.items():
                if 'avg' in name and isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

model.to(device)
optimizer_to(optimizer,device)

accum_step =10

def ANLS(pred,answers):
    if answers[0] is not None:
        scores = []
        for ans in answers:
            ed = editdistance.eval(ans.lower(),pred.lower())
            NL = ed/max(len(ans),len(pred))
            scores.append(1-NL if NL<0.5 else 0)
        return [max(scores)]
    return []

model.train()
best_score=0
log=[]
no_improve=0
for epoch in range(start_epoch,200):  # loop over the dataset multiple times
   for idx, batch in enumerate(dataloader):
        if start_idx>0:
            idx+=start_idx
            if idx>=len(dataloader):
                break
        if idx%500==0 and (idx!=0 or epoch!=0):
            torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(), 'epoch':epoch, 'idx':idx},'state_latest{}.pth'.format(experiment))
            print('saved')
        print('Train e:{}, {}/{}'.format(epoch,idx,len(dataloader)),end='\r')
        # get the inputs;
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        bbox = batch["bbox"].to(device)
        image = batch["image"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # zero the parameter gradients
        if idx%accum_step==1:
            optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                       bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss / accum_step
        #print("Loss:", loss.item())
        loss.backward()

        if idx%accum_step==0:
            optimizer.step()

   start_idx=-1

    
   #for idx, batch in enumerate(valid_dataloader):
   scores=[]
   print('Validating')
   for idx,instance in enumerate(valid_images):
       print('Valid {}/{}'.format(idx,len(valid_images)),end='\r')
       pred = run(instance['image'],instance['question'])
       scores+=ANLS(pred,instance['answers'])

   final_score = np.mean(scores)
   print('e {}, ANLS: {}'.format(epoch,final_score))
   if final_score>best_score:
       best_score = final_score
       torch.save({'state_dict':model.state_dict(), 'epoch':epoch, 'ANLS':final_score},'state_best{}.pth'.format(experiment))
       print('saved best')
       no_improve=0
       if epoch%2==1:
            for g in optimizer.param_groups:
               g['lr'] *= 0.1
            print('drop LR')
   elif epoch>3:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        no_improve+=1
        if no_improve>1:
            print('No improvement for 2 epochs. Ending training')
            break

   log.append(final_score)
   with open('log{}.json'.format(experiment),'w') as f:
       json.dump(log,f)

