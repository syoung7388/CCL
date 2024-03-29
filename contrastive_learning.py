import torch 
import json 
import argparse 
import os 
import random
import torch.backends.cudnn as cudnn
import numpy as np
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from make_dataset import ContrastiveDataset, make_classes
from torch.utils.data import DataLoader
from models import TLM, ILM
import torch.nn as nn
from tqdm import tqdm
import time

# ===================================================
# Note that in this code, 
# lt = labeled text (clean transcript)
# rt = recognized text (noisy ASR transcipt)
# tlm = reference network 
# ilm = inference network
# ===================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='./datasets/slurp', type=str, help='dataset.json path')
    parser.add_argument("--model_name", default='roberta-base', type=str, help='model_name')
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str, help='tokenizer_name')
    parser.add_argument("--target", default='google', type=str, help='google or wave2vec2.0 or hubert')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden_size')
    parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')
    parser.add_argument('--ckpt', type=str, help='ckpt: save path (model pt, train informations)')  
    parser.add_argument('--max_length', type=int, default=64, help='max_len')
    parser.add_argument('--lambda1', type=float, default=0.3, help='lambda1: cls')
    parser.add_argument('--lambda2', type=float, default=0.7, help='lambda2: word')
    parser.add_argument('--seed', type=int, default=42)  
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--duprm', type=int, default=0)
    parser.add_argument('--warmup', type=float, default=0.3)
    args =  parser.parse_args()
    return args

def get_word_contrastive_loss(tlm_output, ilm_output, lt_lt_mask, lt_rt_mask, targ_mask, pred_mask):
    targ_similarity = torch.matmul(tlm_output, tlm_output.permute(0, 2, 1))/scale
    pred_similarity = torch.matmul(tlm_output, ilm_output.permute(0, 2, 1))/scale
    b, q, k = targ_similarity.size()
    targ = m(targ_similarity.view(b, -1)).view(b, q, k) #negative = batch all sample
    pred = m(pred_similarity.view(b, -1)).view(b, q, k)  #negative = batch all sample
    targ, pred = -torch.log(targ), -torch.log(pred)
    targ_loss = targ.masked_select(lt_lt_mask).mean()
    pred_loss = pred.masked_select(lt_rt_mask).mean()
    loss = targ_loss + pred_loss 
    return {
        'targ_similarity_map': targ_similarity,
        'preds_similarity_map': pred_similarity,
        'loss': loss
    }

def get_cls_contrastive_loss(tlm_output, ilm_output):
    tlm_output = tlm_output[:, 0, :]
    ilm_output = ilm_output[:, 0, :]
    N = tlm_output.size(0)
    cls_similarity = torch.matmul(tlm_output, ilm_output.t())/scale
    sim_similarity = m(cls_similarity)
    mask = torch.eq(torch.eye(N), 1).to(sim_similarity.device) #diag
    sim_similarity = sim_similarity.masked_select(mask)
    loss = -torch.log(sim_similarity).mean()
    return {
        "cls_similarity_map": cls_similarity,
        "loss":loss
    }



def validation(e):
    tlm.eval()
    ilm.eval()
    epoch_valid_loss = 0.0
   
    with torch.no_grad():
        with tqdm(total=len(valid_dataloader), desc ='Validation') as vbar:
            for idx, data in enumerate(valid_dataloader):
        
                lt_input_ids, lt_token_type_ids, lt_attention_mask = data['lt_input_ids'].to(device), data['lt_token_type_ids'].to(device), data['lt_attention_mask'].to(device)
                rt_input_ids, rt_token_type_ids, rt_attention_mask = data['rt_input_ids'].to(device), data['rt_token_type_ids'].to(device), data['rt_attention_mask'].to(device)
                lt_lt_mask, lt_rt_mask = data['lt_lt_mask'].to(device), data['lt_rt_mask'].to(device) 
                tlm_output = tlm(lt_input_ids, lt_token_type_ids, lt_attention_mask)
                ilm_output = ilm(rt_input_ids, rt_token_type_ids, rt_attention_mask)
                cls_results = get_cls_contrastive_loss(tlm_output, ilm_output)
                word_results = get_word_contrastive_loss(tlm_output, ilm_output, lt_lt_mask, lt_rt_mask, lt_attention_mask, rt_attention_mask)
        
                cls_loss = cls_results['loss']
                word_loss = word_results['loss']
                tot_loss = args.lambda1*cls_loss + args.lambda2*word_loss
                epoch_valid_loss += tot_loss.item()
                valid_results['valid_iter'].append(tot_loss.item())
                valid_results['valid_cls_iter'].append(cls_loss.item())
                valid_results['valid_word_iter'].append(word_loss.item())
                vbar.update(1)
                
    epoch_valid_loss /= len(valid_dataloader)
    valid_results['valid_epoch'].append(epoch_valid_loss)
    print(f'[valid loss]:{epoch_valid_loss:.5f}') 
    return 



if __name__ == "__main__":
    

    start_time = time.time()

    #[Args] 
    args = parse_args()
    print(str(args))

    #[Gpus] 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #[Seed]
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


    #[Save path] 
    path = f'./results/contrastive/{args.ckpt}'
    if not os.path.exists(path):
        os.makedirs(path+'/models/tlm')
        os.makedirs(path+'/models/ilm')

    #[Load data]
    with open(args.dataset + f'/{args.target}_datas.json', 'r') as f:
        datasets = json.load(f)
    classes = make_classes(args, datasets)
    classnum = len(classes)
    print(f'[class num] {classnum}')
    print(classes)

    #[Max length]
    max_len = 0
    tot_data = datasets['train'] + datasets['devel'] + datasets['test']
    for t in tot_data:
        lt = t['golden']
        lt = lt.split(' ')
        max_len = max(len(lt), max_len)
        rt = t[args.target]['sentence']
        rt = rt.split(' ')
        max_len = max(len(rt), max_len)
    max_len = max_len+10
    print("MAX LEN")
    print(max_len)
    args.max_length = max_len
    
    #[Data]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    train_data = datasets['train']
    train_dataset = ContrastiveDataset(tokenizer, train_data, args.max_length, args.target, 'train', classes, args.duprm) 
    valid_data = datasets['devel']
    valid_dataset = ContrastiveDataset(tokenizer, valid_data, args.max_length, args.target, 'devel', classes, args.duprm) 
    print(f'[dataset size] train: {len(train_dataset)}, devel: {len(valid_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True) #set to True to have the data reshuffled at every epoch
    print(f'[dataloader size] train: {len(train_dataloader)}, devel: {len(valid_dataloader)}')

    #[Model]
    tlm = TLM(args).to(device)
    ilm = ILM(args).to(device)

    #[Train settings]
    model_params = list(tlm.parameters())+list(ilm.parameters())
    optimezer = AdamW(model_params, lr = args.lr, betas = (0.9, 0.999), eps = 1e-8)
    m = nn.Softmax(-1)
    mse_fnc = nn.MSELoss(reduction='mean')
    scale = torch.sqrt(torch.FloatTensor([args.hidden_size])).to(device)
    total_steps = len(train_dataloader)*args.epochs
    num_warmup_steps = int(total_steps*(args.warmup))
    scheduler = get_linear_schedule_with_warmup(optimezer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    print(f'[step] total:{total_steps}, warmup: {num_warmup_steps}')
    train_results = {'train_epoch':[], 'train_iter':[], 'train_cls_iter':[], 'train_word_iter':[]}
    valid_results = {'valid_epoch':[], 'valid_iter':[], 'valid_cls_iter':[], 'valid_word_iter':[]}

    #[Contrastive learning]   
    for e in range(1, args.epochs+1):
        print(f'[epoch]: {e}')
        tlm.train()
        ilm.train()
        epoch_train_loss = 0.0
        with tqdm(total=len(train_dataloader), desc='Training') as pbar:
            for idx, data in enumerate(train_dataloader):
                lt_input_ids, lt_token_type_ids, lt_attention_mask = data['lt_input_ids'].to(device), data['lt_token_type_ids'].to(device), data['lt_attention_mask'].to(device)
                rt_input_ids, rt_token_type_ids, rt_attention_mask = data['rt_input_ids'].to(device), data['rt_token_type_ids'].to(device), data['rt_attention_mask'].to(device)
                label, lt_lt_mask, lt_rt_mask = data['label'].to(device), data['lt_lt_mask'].to(device), data['lt_rt_mask'].to(device) 
                tlm_output = tlm(lt_input_ids, lt_token_type_ids, lt_attention_mask)
                ilm_output = ilm(rt_input_ids, rt_token_type_ids, rt_attention_mask)
                cls_results = get_cls_contrastive_loss(tlm_output, ilm_output)
                word_results = get_word_contrastive_loss(tlm_output, ilm_output, lt_lt_mask, lt_rt_mask, lt_attention_mask, rt_attention_mask)
                cls_loss = cls_results['loss']
                word_loss = word_results['loss']
                tot_loss = args.lambda1*cls_loss + args.lambda2*word_loss
                optimezer.zero_grad()
                tot_loss.backward()
                optimezer.step()
                scheduler.step()
                epoch_train_loss += tot_loss.item()
                train_results['train_iter'].append(tot_loss.item())
                train_results['train_cls_iter'].append(cls_loss.item())
                train_results['train_word_iter'].append(word_loss.item())
                pbar.update(1)
        epoch_train_loss /= len(train_dataloader)
        train_results['train_epoch'].append(epoch_train_loss)
        print(f'[train loss]:{epoch_train_loss:.5f}')
        torch.save(tlm.state_dict(), path+f'/models/tlm/e={e}.pt')
        torch.save(ilm.state_dict(), path+f'/models/ilm/e={e}.pt')

        validation(e)

    #[Save result and info]
    with open(path+'/train_results.json', 'w') as f:
        json.dump(train_results, f, indent='\t')
    with open(path+'/valid_results.json', 'w') as f:
        json.dump(valid_results, f, indent='\t')
    end_time = time.time()
    info = open(path+'/info.txt', 'w')
    info.write(f'train seconds: {end_time - start_time}\n')
    info.write(f'cuda version: {torch.version.cuda}\n')
    info.write(f'torch version: {torch.__version__}\n')
    info.write(f'[dataset size] train: {len(train_dataset)}, devel: {len(valid_dataset)}\n')
    info.write(str(args)+'\n')
    
       


