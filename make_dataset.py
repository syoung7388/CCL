
from doctest import Example
from re import L
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import pandas as pd

def make_classes(args, datasets): 
    train, valid, test = datasets['train'], datasets['devel'], datasets['test']
    train = [(s['scenario'], s['action']) for s in train]
    valid = [(s['scenario'], s['action']) for s in valid]
    test = [(s['scenario'], s['action']) for s in test]
    tot_data = train + valid + test 
    tot_data = list(set(tot_data))
    tot_data.sort()
    classes = {v:k for k, v in enumerate(tot_data)}
    return classes


    

def get_examples(tokenizer, datas, max_length):
    text_examples = []            
    for  text, label in datas:
        tokenized_text = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length, return_attention_mask = True)
        example = {
            "input_ids": torch.tensor(tokenized_text["input_ids"]),
            "token_type_ids": torch.tensor([0 for _ in range(len(tokenized_text["input_ids"]))]),
            "attention_mask": torch.tensor(tokenized_text["attention_mask"]), 
            "label": torch.tensor(label)
        }
        text_examples.append(example)
    return text_examples


def get_paired_examples(tokenizer, text,  max_length):
    text_examples = []
    with tqdm(total=len(text), desc='tokenizer') as pbar:
        for lt, rt, label, pos_token_pair in text:
            lt_tokenized_text = tokenizer(lt, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length, return_attention_mask = True)
            rt_tokenized_text = tokenizer(rt, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length, return_attention_mask = True)
            example = {
                "lt_input_ids": torch.tensor(lt_tokenized_text["input_ids"]),
                "lt_token_type_ids": torch.tensor([0 for _ in range(len(lt_tokenized_text["input_ids"]))]),
                "lt_attention_mask": torch.tensor(lt_tokenized_text["attention_mask"]),
                "rt_input_ids": torch.tensor(rt_tokenized_text["input_ids"]),
                "rt_token_type_ids": torch.tensor([0 for _ in range(len(rt_tokenized_text["input_ids"]))]),
                "rt_attention_mask": torch.tensor(rt_tokenized_text["attention_mask"]),
                "label": torch.tensor(label)
            }
            text_examples.append(example)
            pbar.update(1)
    return text_examples


def get_attn_examples(tokenizer, text,  max_length):
    text_examples = []
    with tqdm(total=len(text), desc='tokenizer') as pbar:
        for id, (lt, rt, label, pos_token_pair) in enumerate(text):
            if len(pos_token_pair) == 0 : continue
            lt_tokenized_text = tokenizer(lt, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length, return_attention_mask = True)
            rt_tokenized_text = tokenizer(rt, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length, return_attention_mask = True)
            lt_rt_mask = torch.zeros(max_length, max_length, dtype=torch.bool)
            lt_lt_mask = torch.zeros(max_length, max_length, dtype=torch.bool)

            for idx in range(len(pos_token_pair)):#(lt_word, rt_word)
                i, j = pos_token_pair[idx]
                if (i >= max_length) or (j >= max_length): continue
                lt_rt_mask[i, j] = True
                lt_lt_mask[i, i] = True

            example = {
                "ID": id, 
                "lt_text":lt, 
                "rt_text":rt, 
                "lt_input_ids": torch.tensor(lt_tokenized_text["input_ids"]),
                "lt_token_type_ids": torch.tensor([0 for _ in range(len(lt_tokenized_text["input_ids"]))]),
                "lt_attention_mask": torch.tensor(lt_tokenized_text["attention_mask"]),
                "rt_input_ids": torch.tensor(rt_tokenized_text["input_ids"]),
                "rt_token_type_ids": torch.tensor([0 for _ in range(len(rt_tokenized_text["input_ids"]))]),
                "rt_attention_mask": torch.tensor(rt_tokenized_text["attention_mask"]),
                "lt_rt_mask": lt_rt_mask,
                "lt_lt_mask": lt_lt_mask,
                "label": torch.tensor(label)
            }
            text_examples.append(example)
            pbar.update(1)
    return text_examples


class ContrastiveDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length, target, dataname, classes, duprm):
        self.data = []
        if duprm: 
            with tqdm(total=len(dataset), desc=f'{dataname}') as pbar:
                for s in dataset:
                    pbar.update(1)
                    label = classes[(s['scenario'], s['action'])]
                    if ((s['golden'], s[target]['sentence'], label, s[target]['pos_token_pair'])) in self.data: continue
                    self.data.append((s['golden'], s[target]['sentence'], label, s[target]['pos_token_pair']))
        else:
            with tqdm(total=len(dataset), desc=f'{dataname}') as pbar:
                for s in dataset:
                    pbar.update(1)
                    label = classes[(s['scenario'], s['action'])]
                    self.data.append((s['golden'], s[target]['sentence'], label, s[target]['pos_token_pair']))       
        self.examples = get_attn_examples(tokenizer, self.data, max_length = max_length)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return 100
        return len(self.examples)


class PairedDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length, target, dataname, classes, duprm):
        self.data = []
        if duprm: #dup remove
            with tqdm(total=len(dataset), desc=f'{dataname}') as pbar:
                for s in dataset:
                    pbar.update(1)
                    label = classes[(s['scenario'], s['action'])]
                    if ((s['golden'], s[target]['sentence'], label, s[target]['pos_token_pair'])) in self.data: continue
                    self.data.append((s['golden'], s[target]['sentence'], label,  s[target]['pos_token_pair']))
        else:
            with tqdm(total=len(dataset), desc=f'{dataname}') as pbar:
                for s in dataset:
                    pbar.update(1)
                    if target not in s.keys(): continue
                    label = classes[(s['scenario'], s['action'])]
                    self.data.append((s['golden'], s[target]['sentence'], label, s[target]['pos_token_pair']))       
                     
        self.examples = get_paired_examples(tokenizer, self.data, max_length = max_length)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return 100
        return len(self.examples)

    

class LTDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length, dup_rm, classes):
        self.data = []
        self.lt = []
        for s in dataset:
            label = classes[(s['scenario'], s['action'])]
            if dup_rm and ((s['golden'], label)) in self.data: continue
            self.data.append((s['golden'], label))  
            self.lt.append(s['golden'])
        self.examples = get_examples(tokenizer, self.data, max_length = max_length)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        #return 100
        return len(self.examples)


class RTDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length, target, dup_rm, classes):
        self.data = []
        self.lt = []
        self.rt = []
        for s in dataset:
            label = classes[(s['scenario'], s['action'])]
            if dup_rm and (s[target]['sentence'], label) in self.data: continue
            self.data.append((s[target]['sentence'], label))
            self.lt.append(s['golden'])
            self.rt.append(s[target]['sentence'])
        self.examples = get_examples(tokenizer, self.data, max_length = max_length)

        
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        #return 100
        return len(self.examples)
            



