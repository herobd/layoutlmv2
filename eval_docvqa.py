# -*- coding: utf-8 -*-
import numpy as np
import editdistance
import sys
from PIL import Image
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


finetuned_checkpoint = sys.argv[1]
outfile = sys.argv[2]

model_checkpoint = "../pairing/cache_huggingface/layoutlmv2-base-uncased"



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




import json
with open('../data/DocVQA/test/test_v1.0.json') as f:
  test_data = json.load(f)
import pandas as pd
test_df = pd.DataFrame(test_data['data'])
test_df.head()
root_dir = '../data/DocVQA/test/'
from datasets import Dataset
test_dataset = Dataset.from_pandas(test_df.iloc[:])
def get_image(examples):
    
  images = [root_dir + image_file for image_file in examples['image']]
  examples['image']=images
  return examples

test_images = test_dataset.map(get_image, batched=True, batch_size=2)

from transformers import LayoutLMv2Processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")



from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

loading = torch.load(finetuned_checkpoint)

model.load_state_dict(loading['state_dict'])


device = "cuda:1" if torch.cuda.is_available() else "cpu"


model.to(device)

def ANLS(pred,answers):
    if answers[0] is not None:
        scores = []
        for ans in answers:
            ed = editdistance.eval(ans.lower(),pred.lower())
            NL = ed/max(len(ans),len(pred))
            scores.append(1-NL if NL<0.5 else 0)
        return [max(scores)]
    return []

model.eval()
results=[]
for idx,instance in enumerate(test_images):
   print('Test {}/{}'.format(idx,len(test_images)))
   pred = run(instance['image'],instance['question'])
   results.append({
       'questionId': instance['questionId'],
       'answer': pred
       })
   print(instance['question']+' :: '+pred)

with open(outfile,'w') as f:
    json.dump(results,f)
