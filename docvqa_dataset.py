import numpy as np
import editdistance

import json
import torch
import pandas as pd
from PIL import Image

from datasets import Dataset
from transformers import LayoutLMv2FeatureExtractor
from transformers import AutoTokenizer
    
#
import transformers
import random

def fuzzy(s1,s2):
    return (editdistance.eval(s1,s2)/((len(s1)+len(s2))/2)) < 0.2
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

class DocVQADataset(torch.utils.data.Dataset):
    def __init__(self,split,load_ocr=False):
        if split=='train':
            with open('../data/DocVQA/train/train_v1.0.json') as f:
              data = json.load(f)
            self.root_dir = '../data/DocVQA/train/'
        else:
            with open('../data/DocVQA/val/val_v1.0.json') as f:
              data = json.load(f)
            self.root_dir = '../data/DocVQA/val/'

        self.load_ocr=load_ocr

        #data.keys()

        print("Dataset name:", data['dataset_name'])
        print("Dataset split:", data['dataset_split'])



        df = pd.DataFrame(data['data'])
        df.head()




        #print("TESTING: only using 10 instances")
        self.dataset = Dataset.from_pandas(df.iloc[:])


        self.feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=not self.load_ocr)
        model_checkpoint = "../pairing/cache_huggingface/layoutlmv2-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#dataset[0]

#len(dataset)


    def load_ocr_words_and_boxes(self,example):
        
      # get a batch of document imagesa
      image_file = example['image']
      image = Image.open(self.root_dir + image_file).convert("RGB")
      
      # resize every image to 224x224 + apply tesseract to get words + normalized boxes
      encoded_inputs = self.feature_extractor([image])
      example['image'] = encoded_inputs.pixel_values[0]

      ocr_file = image_file.replace('documents/','ocr_results/').replace('.png','.json')
      with open(self.root_dir + ocr_file) as f:
          ocr = json.load(f)
      words=[]
      boxes=[]
      for doc in ocr['recognitionResults']:
          for line in doc['lines']:
              for w in line['words']:
                  points=w['boundingBox']
                  words.append(w['text'])
                  x1 = min(points[0::2])
                  x2 = max(points[0::2])
                  y1 = min(points[1::2])
                  y2 = max(points[1::2])
                  boxes.append(normalize_bbox((x1,y1,x2,y2),image.width,image.height))
      example['words'] = words
      example['boxes'] = boxes

      return example

    def get_ocr_words_and_boxes(self,example):
        
      # get a batch of document imagesa
      image_file = example['image']
      image = Image.open(self.root_dir + image_file).convert("RGB")
      
      # resize every image to 224x224 + apply tesseract to get words + normalized boxes
      encoded_inputs = self.feature_extractor([image])


      example['image'] = encoded_inputs.pixel_values[0]
      example['words'] = encoded_inputs.words[0]
      example['boxes'] = encoded_inputs.boxes[0]

      return example

    
    # source: https://stackoverflow.com/a/12576755
    def subfinder(self, words_list, answer_list):  
        matches = []
        start_indices = []
        end_indices = []
        for idx, i in enumerate(range(len(words_list))):
            #if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            if len(words_list[i:i+len(answer_list)])==len(answer_list) and all(fuzzy(words_list[i+j],answer_list[j]) for j in range(len(answer_list))):
                matches.append(answer_list)
                start_indices.append(idx)
                end_indices.append(idx + len(answer_list) - 1)
        if matches:
          return matches[0], start_indices[0], end_indices[0]
        else:
          return None, 0, 0

    def encode_dataset(self,example, max_length=512):
      # take a batch 
      questions = example['question']
      words = [w for w in example['words']] #handles numpy and list
      boxes = example['boxes']

      # encode it
      encoding = self.tokenizer([questions], [words], [boxes], max_length=max_length, padding="max_length", truncation=True,return_tensors="pt")
      batch_index=0
      input_ids = encoding.input_ids[batch_index].tolist()

      # next, add start_positions and end_positions
      start_positions = []
      end_positions = []
      answers = example['answers']
      #print("Batch index:", batch_index)
      cls_index = input_ids.index(self.tokenizer.cls_token_id)
      # try to find one of the answers in the context, return first match
      words_example = [word.lower() for word in words]
      for answer in answers:
        match, word_idx_start, word_idx_end = self.subfinder(words_example, answer.lower().split())
        #if match:
        #  break
        # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
        if not match and len(answer)>1:
            for i in range(len(answer)):
              # drop the ith character from the answer
              answer_i = answer[:i] + answer[i+1:]
              # check if we can find this one in the context
              match, word_idx_start, word_idx_end = self.subfinder(words_example, answer_i.lower().split())
              if match:
                break
        # END OF EXPERIMENT 
        if match:
          sequence_ids = encoding.sequence_ids(batch_index)
          # Start token index of the current span in the text.
          token_start_index = 0
          while sequence_ids[token_start_index] != 1:
              token_start_index += 1

          # End token index of the current span in the text.
          token_end_index = len(input_ids) - 1
          while sequence_ids[token_end_index] != 1:
              token_end_index -= 1
          
          word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index+1]

          hit=False
          for id in word_ids:
            if id == word_idx_start:
              start_positions.append(token_start_index)
              hit=True
              break
            else:
              token_start_index += 1

          if not hit:
              continue
    
          hit=False
          for id in word_ids[::-1]:
            if id == word_idx_end:
              end_positions.append(token_end_index)
              hit=True
              break
            else:
              token_end_index -= 1

          if not hit:
              end_positions.append(token_end_index)
          
          #print("Verifying start position and end position:")
          #print("True answer:", answer)
          #start_position = start_positions[-1]
          #end_position = end_positions[-1]
          #reconstructed_answer = tokenizer.decode(encoding.input_ids[batch_index][start_position:end_position+1])
          #print("Reconstructed answer:", reconstructed_answer)
          #print("-----------")
        
        #else:
          #print("Answer not found in context")
          #print("-----------")
          #start_positions.append(cls_index)
          #end_positions.append(cls_index)

      if len(start_positions)==0:
          return None
    
      ans_i = random.randrange(len(start_positions))

      encoding = {
              'input_ids': encoding['input_ids'],
              'attention_mask': encoding['attention_mask'],
              'token_type_ids': encoding['token_type_ids'],
              'bbox': encoding['bbox'],
              }
      
      encoding['image'] = torch.LongTensor(example['image'].copy())
      encoding['start_position'] = torch.LongTensor([start_positions[ans_i]])
      encoding['end_position'] = torch.LongTensor([end_positions[ans_i]])

      return encoding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        data = self.dataset[index]
        if self.load_ocr:
            data = self.load_ocr_words_and_boxes(data)
        else:
            name = data['docId']
            try:
                with open("data/{}.bin".format(name),"rb") as f:
                    data['image'] = np.load(f)
                    data['words'] = np.load(f)
                    data['boxes'] = np.load(f)
                #print('loaded data/{}.bin'.format(name))
            except:
                data = self.get_ocr_words_and_boxes(data)
                with open("data/{}.bin".format(name),"wb") as f:
                    np.save(f,data['image'])
                    np.save(f,data['words'])
                    np.save(f,data['boxes'])
                #print('saved data/{}.bin'.format(name))

        data = self.encode_dataset(data)

        if data is None:
            return self.__getitem__((index+1)%len(self))

        return data

def collate(data):
    return {
            'input_ids': torch.cat([d['input_ids'] for d in data],dim=0),
            'attention_mask': torch.cat([d['attention_mask'] for d in data],dim=0),
            'token_type_ids': torch.cat([d['token_type_ids'] for d in data],dim=0),
            'bbox': torch.cat([d['bbox'] for d in data],dim=0),
            'image': torch.stack([d['image'] for d in data],dim=0),
            'start_positions': torch.cat([d['start_position'] for d in data],dim=0),
            'end_positions': torch.cat([d['end_position'] for d in data],dim=0),
            }

