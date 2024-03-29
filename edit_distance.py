

import json
import torch
from transformers import AutoTokenizer
from torchaudio.functional import edit_distance
from tqdm import tqdm
import argparse 
import os 
from jiwer import wer

def convert_id_to_token(ids):
    token = tokenizer._convert_id_to_token(ids)
    if 'Ġ' in token:
        token = token[1:]
    return token

def one_sample_eitsitance(tokenizer, lt, rt):
    toknized_lt = tokenizer(lt)
    toknized_rt = tokenizer(rt)
    lt_ch = [0]*len(toknized_lt)
    rt_ch = [0]*len(toknized_rt)
    cnt = 1
    pos_token_pair = []
    pos_word_pair  = []       

    for i in range(len(lt_ch)):
        for j in range(len(rt_ch)):
            if rt_ch[j] == 0 and toknized_lt[i] == toknized_rt[j]:
                lt_ch[i] = cnt 
                rt_ch[j] = cnt 
                cnt += 1
                pos_token_pair.append((i, j))                        
                pos_word_pair.append((convert_id_to_token(toknized_lt[i]), convert_id_to_token(toknized_rt[j])))
                break

    for i in range(len(lt_ch)):
        if lt_ch[i] != 0: continue
        lt_n = lt_ch[i-1]
        if lt_n == 0: continue
        lt_word = convert_id_to_token(toknized_lt[i])
        rt_words = []
        rt_n = rt_ch.index(lt_n)+1 
        for j in range(rt_n, len(rt_ch)):
            if rt_ch[j] == 0:
                rt_word = convert_id_to_token(toknized_rt[j])
                ed = edit_distance(lt_word, rt_word)
                if ed >= 10: continue
                rt_words.append((ed, j, rt_word))
            else:
                break

        if len(rt_words) == 0: continue
        rt_words = sorted(rt_words, key=lambda x:x[0])
        pos_token_pair.append((i, rt_words[0][1]))
        pos_word_pair.append((lt_word, rt_words[0][-1]))
        
    return


def get_editdistance(tokenizer, target, threshold, datas):
    result_data = []
    nan_result = []
    WER = 0.0
    with tqdm(total=len(datas), desc ='train_data') as bar:
        for idx, data in enumerate(datas):
            if target not in data.keys(): continue
            lt, rt = data['golden'], data[target]['sentence']
            W = wer(lt, rt)
            WER += W
            if rt =='': 
                nan_result.append({'lt':lt, 'intent':data['scenario']})
                continue
            toknized_lt = tokenizer(lt)
            toknized_lt = toknized_lt['input_ids']
            toknized_rt = tokenizer(rt)
            toknized_rt = toknized_rt['input_ids']
            lt_ch = [0]*len(toknized_lt)
            rt_ch = [0]*len(toknized_rt)
            cnt = 1
            pos_token_pair = []
            pos_word_pair  = []


            for i in range(len(lt_ch)):
                for j in range(len(rt_ch)):
                    if rt_ch[j] == 0 and toknized_lt[i] == toknized_rt[j]:
                        lt_ch[i] = cnt 
                        rt_ch[j] = cnt 
                        cnt += 1
                        pos_token_pair.append((i, j))                        
                        pos_word_pair.append((convert_id_to_token(toknized_lt[i]), convert_id_to_token(toknized_rt[j])))
                        break

            for i in range(len(lt_ch)):
                if lt_ch[i] != 0: continue
                lt_n = lt_ch[i-1]
                if lt_n == 0: continue
                lt_word = convert_id_to_token(toknized_lt[i])
                rt_words = []
                rt_n = rt_ch.index(lt_n)+1 
                for j in range(rt_n, len(rt_ch)):
                    if rt_ch[j] == 0:
                        rt_word = convert_id_to_token(toknized_rt[j])
                        ed = edit_distance(lt_word, rt_word)
                        if ed >= threshold: continue
                        rt_words.append((ed, j, rt_word))
                    else:
                        break
                if len(rt_words) == 0: continue
                rt_words = sorted(rt_words, key=lambda x:x[0])
                pos_token_pair.append((i, rt_words[0][1]))
                pos_word_pair.append((lt_word, rt_words[0][-1]))

            

            result = {
                'id': data['id'],
                'file': data['file'],
                'golden': data['golden'],
                'scenario': data['scenario'],
                'action': data['action'],
                target: {
                    'sentence': data[target]['sentence'],
                    'pos_token_pair': pos_token_pair,
                    'pos_word_pair': pos_word_pair
                }
            }
            result_data.append(result)
            bar.update(1)
    #return WER
    return nan_result, result_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="", type=str, help='loda dataset path')
    parser.add_argument("--dataset", default="timers", type=str, help='loda dataset path')
    parser.add_argument("--target", default='hubert', type=str, help='google or wave2vec2.0 or hubert')
    parser.add_argument("--threshold", default=10, type=int, help='word edit count treshold')
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str, help='tokenizer_name')
    parser.add_argument("--ckpt", default='timers', type=str, help='ckpt')
    args =  parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name) 

    #load data
    with open(args.dataset_path, 'r') as f:
        datasets = json.load(f)

    
    test_data = []
    if args.target != 'hubert' and args.dataset == 'slurp':
        testname = 'google_test' if args.target == 'google'  else 'wave2vec_test'
        for data in datasets[testname]:
            test_data += data
    else:
        test_data = datasets['test']

    train_n, test_n = len(datasets['train']),  len(test_data)
    print(f'[LEN] train: {train_n}, test: {test_n}')
    train_data = datasets['train']
    train_nan, train_data = get_editdistance(tokenizer, args.target, args.threshold, train_data)
    if 'devel' in datasets.keys():
        devel_nan, devel_data = get_editdistance(tokenizer=tokenizer, target=args.target, threshold=args.threshold, datas=datasets['devel'])
    test_nan, test_data = get_editdistance(tokenizer=tokenizer, target=args.target, threshold=args.threshold, datas=test_data)
    train_n, test_n = len(train_data),  len(test_data)
    print(f'[LEN] train: {train_n},  test: {test_n}')
    
    print(f'[NAN] train: {len(train_nan)}, test: {len(test_nan)}')

    if 'devel' in datasets.keys(): 
        results = {
            'train': train_data,
            'devel': devel_data,
            'test': test_data
        }
        nan_results= {
            'train': train_nan,
            'devel': devel_nan,
            'test': test_nan    
        }
    else:
        results = {
            'train': train_data,
            'test': test_data
        }
        nan_results= {
            'train': train_nan,
            'test': test_nan    
        }

    if not os.path.exists(f'./datasets/{args.ckpt}'):
        os.makedirs(f'./datasets/{args.ckpt}')

    with open(f'./datasets/{args.ckpt}/{args.target}_datas.json', 'w') as f:
        json.dump(results, f, indent='\t')





