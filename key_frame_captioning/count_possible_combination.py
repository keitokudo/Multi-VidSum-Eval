import pickle
import json
import argparse
import yaml
import os

import random
import copy
import numpy as np

# torch
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

# our implements
from Dataset import ANetDataset
from FrameDetectDataset import  FrameANetDataset
from tokenizer import get_tokenizer
from models.baseline import BaselineModel
from evaluator import Evaluator
from criterion import get_criterion


def create_golds_captions(dataloader):
    golds = defaultdict(list)
    gold_captions = defaultdict(list)
    gold_ann_secs = defaultdict(list)
    gold_segment_timestamps = defaultdict(list)

    with torch.no_grad():
        for b_resnet, b_bni, b_sent_ids, _, b_frame_masks, b_sent_mask, b_vid, _, (b_original_gold_labels, b_start_idx, _, b_original_frame_cnt), (b_ann_secs, b_captions), b_gold_captions, _, (b_start_sec, b_end_sec) in dataloader:
            for i in range(b_resnet.size(0)):
                golds[b_vid[i]].append(
                    b_original_gold_labels[i][:b_original_frame_cnt[i]].detach().cpu().numpy().tolist())
                gold_captions[b_vid[i]].append(b_gold_captions[i])
                gold_segment_timestamps[b_vid[i]].append([b_start_sec[i], b_end_sec[i]])
                gold_ann_secs[b_vid[i]].append(b_ann_secs[i])
                    
    return golds, gold_captions, gold_segment_timestamps, gold_ann_secs

def main(args):
    #set config
    eval_config = yaml.safe_load(open(args.eval_config_file))
    model_config = yaml.safe_load(open(args.model_config_file))
    tokenizer, vocab, pad_token_id = get_tokenizer(
        model_config["model"]["language_model"], eval_config["dataset_file"])
    debug=False
    use_sub_movie=True
    use_only_movie_exist_vid=True
    num_workers=6
    seed=1

    #set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    #load dataset
    oracle_dev_dataset = ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="validation", use_sub_movie=use_sub_movie, use_only_movie_exist_vid=use_only_movie_exist_vid)

    oracle_test_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="testing", use_sub_movie=use_sub_movie, use_only_movie_exist_vid=use_only_movie_exist_vid)        

    dev_dataloader = DataLoader(oracle_dev_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=oracle_dev_dataset.collate_fn)

    oracle_test_dataloader = DataLoader(oracle_test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=oracle_test_dataset.collate_fn)                             


    test_golds, test_captions, test_gold_timestamps, test_ann_secs = create_golds_captions(oracle_test_dataloader)
    dev_golds, dev_captions, dev_gold_timestamps, dev_ann_secs = create_golds_captions(dev_dataloader)

    # devとtestのgoldsをまとめる
    golds = defaultdict(list)
    test_vids_set = set(test_golds.keys())
    dev_vids_set = set(dev_golds.keys())
    gold_timestamps = defaultdict(list)
    gold_captions = defaultdict(list)
    gold_ann_secs = defaultdict(list)


    for vid in dev_vids_set | test_vids_set:
        if vid in dev_vids_set:
            golds[vid].extend(dev_golds[vid])
            gold_timestamps[vid].extend(dev_gold_timestamps[vid])
            gold_captions[vid].extend(dev_captions[vid])
            gold_ann_secs[vid].extend(dev_ann_secs[vid])
        if vid in test_vids_set:
            golds[vid].extend(test_golds[vid])
            gold_timestamps[vid].extend(test_gold_timestamps[vid])
            gold_captions[vid].extend(test_captions[vid])
            gold_ann_secs[vid].extend(test_ann_secs[vid])
        
    pickle_data = dict()
    pickle_data["golds"] = copy.deepcopy(golds)
    pickle_data["gold_timestamps"] = copy.deepcopy(gold_timestamps)
    pickle_data["gold_captions"] = copy.deepcopy(gold_captions)
    pickle_data["gold_ann_secs"] = copy.deepcopy(gold_ann_secs)

    pickle.dump(pickle_data, open('../gold_exist_vid.pickle', 'wb'))
    json.dump(pickle_data, open('../gold_exist_vid.json', 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config_file", help="", default='../config/AAAI2021/train_bilstm_config.yml', type=str)
    parser.add_argument("--model_config_file", help="", default='../config/AAAI2021/model_bilstm_config.yml', type=str)
    #parser.add_argument("--port", help="Specify clip-rewards server's port number", type=int, required=True)
    #parser.add_argument("--karpathy_path", help="Specify karpathy_tset_iamges.txt path", type=Path, required=True)
    #parser.add_argument("--output", help="Specify output file name", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
