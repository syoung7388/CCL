
import torch 
import json 
import argparse 
import os 
import random
import torch.backends.cudnn as cudnn
import numpy as np
from transformers import AutoTokenizer,  AdamW, get_linear_schedule_with_warmup
from models import TargetNet, InferNet
from make_dataset import PairedDataset, make_classes
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import time



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='./datasets/slurp', type=str, help='dataset.json path')
    parser.add_argument("--model_name", default='roberta-base', type=str, help='model_name')
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str, help='tokenizer_name')
    parser.add_argument("--target", default='google', type=str, help='google or wave2vec2.0 or hubert')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning_rate') 
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden_size')
    parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--ckpt', type=str, help='ckpt: save path (model pt, train informations)')
    parser.add_argument('--tlm_path', type=str, default='/test/models/tlm/e=5.pt' , help='tlm model dir')   
    parser.add_argument('--ilm_path', type=str, default='/test/models/ilm/e=5.pt', help='ilm model dir')   
    parser.add_argument('--max_length', type=int, default=64, help='max_len')
    parser.add_argument('--seed', type=int, default=42)  
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model_load', type=int, default=1)
    parser.add_argument('--lambda1', type=float, default=0.3)
    parser.add_argument('--lambda2', type=float, default=0.7)
    parser.add_argument('--duprm', type=int, default=0)
    parser.add_argument('--linear', type=int, default=0)
    args =  parser.parse_args()
    return args




if __name__ == "__main__":

    start_time = time.time()

    #[Args] 
    args = parse_args()
    print(str(args))

    #[Gpus] 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #[Seed] 
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    #[Load data]
    with open(args.dataset + f'/{args.target}_datas.json', 'r') as f:
        datasets = json.load(f)
    n_train, n_devel, n_test = len(datasets['train']), len(datasets['devel']), len(datasets['test'])
    
    #[Class]
    classes = make_classes(args, datasets)
    classnum = len(classes)
    print(classes)
    print(f'[class num] {classnum}')


    #[Max length]
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
    train_dataset = PairedDataset(tokenizer, datasets['train'], args.max_length, args.target, 'train', classes, args.duprm) 
    valid_data = datasets['devel']
    valid_dataset = PairedDataset(tokenizer, valid_data, args.max_length, args.target, 'devel', classes, 0)
    test_dataset = PairedDataset(tokenizer, datasets['test'], args.max_length, args.target, 'test', classes, 0)
    print(f'[dataset size] train: {len(train_dataset)}, devel: {len(valid_dataset)}, test: {len(test_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) 
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f'[dataloader size] train: {len(train_dataloader)}, devel: {len(valid_dataloader)}, test:{len(test_dataloader)}')   
    

    #[Save path] 
    path = f'./results/finetuning/{args.ckpt}'
    if not os.path.exists(path):
        os.makedirs(path+'/models/targetnet')
        os.makedirs(path+'/models/infernet')

    #[Model]
    targetnet = TargetNet(args, classnum).to(device)
    infernet = InferNet(args, classnum).to(device)
 
    #[Train settings]
    target_optimizer = AdamW(targetnet.parameters(), lr = args.lr , betas = (0.9, 0.999), eps = 1e-8)
    target_total_steps = len(train_dataloader)*args.epochs
    target_num_warmup_steps = int(target_total_steps*(1/3))
    target_scheduler = get_linear_schedule_with_warmup(target_optimizer, num_warmup_steps=target_num_warmup_steps, num_training_steps=target_total_steps)
    print(f'[targetnet step] total: {target_total_steps}, warmup: {target_num_warmup_steps}')

    infer_optimizer = AdamW(infernet.parameters(), lr = args.lr, betas = (0.9, 0.999), eps = 1e-8)
    infer_total_steps = len(train_dataloader)*args.epochs 
    infer_num_warmup_steps = int(infer_total_steps*(1/3))
    infer_scheduler = get_linear_schedule_with_warmup(infer_optimizer, num_warmup_steps=infer_num_warmup_steps, num_training_steps=infer_total_steps)
    print(f'[infernet step] total: {infer_total_steps}, warmup: {infer_num_warmup_steps}')

    mse_loss_fnc = nn.MSELoss(reduction='mean')
    cel_fnc = nn.CrossEntropyLoss()

    #[Save result] 
    target_results = {'train_loss':[], 'valid_loss':[], 'test_acc':[]}
    infer_results = {'train_ce':[], 'train_mse':[], 'train_loss':[], 'valid_ce':[], 'valid_mse':[], 'valid_loss':[], 'test_acc':[]}
    tot_trg_train, tot_trg_valid = 0.0, 0.0
    tot_inf_train, tot_inf_valid = 0.0, 0.0
    tot_inf_ce_train, tot_inf_ce_valid, tot_inf_mse_train, tot_inf_mse_valid = 0.0, 0.0, 0.0, 0.0 
    train_iter = 1
    valid_iter = 1

    for e in range(1, args.epochs+1):
        print(f'============ [epcochs] {e}/{args.epochs} ============')

        #[Consistency learning] 
        targetnet.train()
        infernet.train()
        with tqdm(total=len(train_dataloader), desc ='Training') as pbar:
            for data in train_dataloader:
                lt_input_ids, lt_token_type_ids, lt_attention_mask = data['lt_input_ids'].to(device), data['lt_token_type_ids'].to(device), data['lt_attention_mask'].to(device)
                rt_input_ids, rt_token_type_ids, rt_attention_mask = data['rt_input_ids'].to(device), data['rt_token_type_ids'].to(device), data['rt_attention_mask'].to(device)
                label = data['label'].to(device)
                #target
                target_result = targetnet(lt_input_ids, lt_token_type_ids, lt_attention_mask, label)
                t_ce_loss = cel_fnc(target_result['last_output'], label)
                target_optimizer.zero_grad()
                t_ce_loss.backward(retain_graph=True)
                target_optimizer.step()
                target_scheduler.step()
                tot_trg_train += t_ce_loss.item()
                target_results['train_loss'].append(tot_trg_train/train_iter)
                #infer 
                infer_result = infernet(rt_input_ids, rt_token_type_ids, rt_attention_mask, label)
                infer_loss = cel_fnc(infer_result['last_output'], label)    
                i_ce_loss = (infer_loss - t_ce_loss)**2
                target_output, infer_output = target_result['last_output'], infer_result['last_output']
                i_mse_loss = mse_loss_fnc(infer_output, target_output)
                i_train_loss = args.lambda1*(i_ce_loss) + args.lambda2*(i_mse_loss)
                infer_optimizer.zero_grad()
                i_train_loss.backward()
                infer_optimizer.step()
                infer_scheduler.step()
                #save
                tot_inf_train += i_train_loss.item()
                infer_results['train_loss'].append(tot_inf_train/train_iter)
                tot_inf_ce_train += infer_loss.item()
                infer_results['train_ce'].append(tot_inf_ce_train/train_iter)
                tot_inf_mse_train += i_mse_loss.item()
                infer_results['train_mse'].append(tot_inf_mse_train/train_iter)
                train_iter += 1
                pbar.update(1)
        target_loss, infer_loss = target_results['train_loss'][-1], infer_results['train_ce'][-1]
        print(f'[train loss] target:{target_loss:.5f}, infer:{infer_loss:.5f}')

        #[Valid] 
        targetnet.eval()
        infernet.eval()
        with torch.no_grad():
            with tqdm(total=len(valid_dataloader), desc ='Validation') as vbar:
                for data in valid_dataloader:
                    lt_input_ids, lt_token_type_ids, lt_attention_mask = data['lt_input_ids'].to(device), data['lt_token_type_ids'].to(device), data['lt_attention_mask'].to(device)
                    rt_input_ids, rt_token_type_ids, rt_attention_mask = data['rt_input_ids'].to(device), data['rt_token_type_ids'].to(device), data['rt_attention_mask'].to(device)
                    label = data['label'].to(device)
                    #target
                    target_result = targetnet(lt_input_ids, lt_token_type_ids, lt_attention_mask, label)
                    t_ce_loss = cel_fnc(target_result['last_output'], label)    
                    tot_trg_valid += t_ce_loss.item()
                    target_results['valid_loss'].append(tot_trg_valid/valid_iter)
                    #infer
                    infer_result = infernet(rt_input_ids, rt_token_type_ids, rt_attention_mask, label) 
                    infer_loss = cel_fnc(infer_result['last_output'], label)     
                    i_ce_loss = (infer_loss - t_ce_loss)**2
                    target_output, infer_output = target_result['last_output'], infer_result['last_output']
                    i_mse_loss = mse_loss_fnc(infer_output, target_output)
                    i_valid_loss = args.lambda1*(i_ce_loss) + args.lambda2*(i_mse_loss)
                    #save
                    tot_inf_valid += i_valid_loss.item()
                    infer_results['valid_loss'].append(tot_inf_valid/valid_iter)
                    tot_inf_ce_valid += infer_loss.item()
                    infer_results['valid_ce'].append(tot_inf_ce_valid/valid_iter)
                    tot_inf_mse_valid += i_mse_loss.item()
                    infer_results['valid_mse'].append(tot_inf_mse_valid/valid_iter)
                    valid_iter += 1
                    vbar.update(1)
        target_loss, infer_loss = target_results['valid_loss'][-1], infer_results['valid_ce'][-1]
        print(f'[valid loss] target:{target_loss:.5f}, infer:{infer_loss:.5f}')

        #[Test]
        targetnet.eval()
        infernet.eval()
        t_test_acc = 0.0
        i_test_acc = 0.0
        with torch.no_grad():
            for data in test_dataloader:
                lt_input_ids, lt_token_type_ids, lt_attention_mask = data['lt_input_ids'].to(device), data['lt_token_type_ids'].to(device), data['lt_attention_mask'].to(device)
                rt_input_ids, rt_token_type_ids, rt_attention_mask = data['rt_input_ids'].to(device), data['rt_token_type_ids'].to(device), data['rt_attention_mask'].to(device)
                label = data['label'].to(device)
                #target rt acc
                target_result = targetnet(rt_input_ids, rt_token_type_ids, rt_attention_mask, label)
                target_preds = torch.argmax(target_result['last_output'])
                t_test_acc += torch.sum(target_preds == label).item()
                #infer rt acc 
                infer_result = infernet(rt_input_ids, rt_token_type_ids, rt_attention_mask, label)
                infer_preds = torch.argmax(infer_result['last_output'])
                i_test_acc += torch.sum(infer_preds == label).item()
        t_test_acc = (t_test_acc/len(test_dataloader))*100 
        i_test_acc = (i_test_acc/len(test_dataloader))*100 
        target_results['test_acc'].append(t_test_acc)
        infer_results['test_acc'].append(i_test_acc)
        print(f'[rt test accuracy] target: {t_test_acc:.5f}%, infer: {i_test_acc:.5f}%')
        torch.save(targetnet.state_dict(), path+f'/models/targetnet/e={e}.pt')
        torch.save(infernet.state_dict(), path+f'/models/infernet/e={e}.pt')


    with open(path+'/target_result.json', 'w') as f:
        json.dump(target_results, f, indent='\t')
    with open(path+'/infer_result.json', 'w') as f:
        json.dump(infer_results, f, indent='\t')

    tot_log = open('./results/tot_result.txt', 'a')
    tot_log.write(str(args)+'\n')
    tot_log.write(f'[test acc] target: {t_test_acc:.5f}%, infer: {i_test_acc:.5f}%\n')
    end_time = time.time()
    info = open(path+'/info.txt', 'w')
    info.write(f'train seconds: {end_time - start_time}\n')
    info.write(f'cuda version: {torch.version.cuda}\n')
    info.write(f'torch version: {torch.__version__}\n')
    info.write(f'[dataset size] train: {len(train_dataset)}, devel: {len(valid_dataset)}, test:{len(test_dataset)}\n')
    info.write(str(args)+'\n')
    
       

