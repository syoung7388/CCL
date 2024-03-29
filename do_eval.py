
import torch 
import json 
import argparse 
import os 
import random
import torch.backends.cudnn as cudnn
import numpy as np
from transformers import AutoTokenizer
from models import InferNet
from make_dataset import  make_classes, PairedDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score,  f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='./datasets/slurp', type=str, help='slurp or timers or fsc or snips')
    parser.add_argument("--model_name", default='roberta-base', type=str, help='model_name')
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str, help='tokenizer_name')
    parser.add_argument("--target", default='google', type=str, help='google or wave2vec2.0')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden_size')
    parser.add_argument('--gpus', type=str, default='1', help='gpu numbers')
    parser.add_argument('--max_length', type=int, default=64, help='max_len')
    parser.add_argument('--seed', type=int, default=42)  
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--our_model', type=str, default="./best_models/slurp_google.pt", help='our_model load dir')
    parser.add_argument('--model_load', type=int, default=0, help='model_load')
    args =  parser.parse_args()
    return args



def testing(dataloader):
    result = {
        'predict':[],
        'label':[]
    }

    our_model.eval()
    with tqdm(total=len(dataloader), desc=f'[TEST]') as pbar:
        for data in dataloader:
            input_ids, token_type_ids, attention_mask = data['rt_input_ids'].to(device), data['rt_token_type_ids'].to(device), data['rt_attention_mask'].to(device)
            label = data['label'].to(device)
            predict = our_model(input_ids, token_type_ids, attention_mask, label)
            predict =  torch.argmax(predict['last_output'], -1)
            predict = predict.cpu().numpy().tolist()
            result['predict'] += predict
            result['label'] += label.cpu().numpy().tolist()
            pbar.update(1)

    f1 = f1_score(result['label'], result['predict'], average='macro')*100
    acc = accuracy_score(result['label'], result['predict'])*100
    print(f'[RESURT]: f1: {f1:.3f}, acc: {acc:.3f}')

    return 
    
    
        





if __name__ == "__main__":

    start_time = time.time()

    # args 
    args = parse_args()

    # gpus 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #seed 
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


    #load data
    with open(args.dataset + f'/{args.target}_datas.json', 'r') as f:
        datasets = json.load(f)

    #class
    classes = make_classes(args, datasets)
    classnum = len(classes)
    print(classes)
    print(f'[class num] {classnum}')


    #max length ch
    max_len = 0
    tot_data = datasets['test'] +datasets['train'] + datasets['devel']
    for t in tot_data:
        lt = t['golden']
        lt = lt.split(' ')
        max_len = max(len(lt), max_len)
        if args.target in t.keys():
            rt = t[args.target]['sentence']
            rt = rt.split(' ')
            max_len = max(len(rt), max_len)
    max_len = max_len+10
    print("MAX LEN")
    print(max_len)
    args.max_length = max_len

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    test_dataset = PairedDataset(tokenizer, datasets['test'], args.max_length, args.target, 'test', classes, 0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    our_model = InferNet(args, classnum).to(device)
    our_model.load_state_dict(torch.load(args.our_model))

    results = testing(test_dataloader)



