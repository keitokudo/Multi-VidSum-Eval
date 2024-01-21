'''
説明
'''
import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ip', '--input_pred', type=path.abspath, help='input file path')
    parser.add_argument(    
        '-ig', '--input_gold', type=path.abspath, help='input file path')
#    parser.add_argument('--log',
#                        default="log.txt", help='Path to log file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)
    pred_data = json.load(open(args.input_pred))
    gold_data = json.load(open(args.input_gold))


    for vid in set(gold_data.keys()) & set(pred_data.keys()):
        for i in range(len(gold_data[vid])):
            start_idx = gold_data[vid][i]["start"]
            end_idx = gold_data[vid][i]["end"]
            for ann_idx in gold_data[vid][i]["ann_secs"]:
                assert ann_idx <= end_idx and start_idx <= ann_idx, f'''{vid} ann_idx:{ann_idx}, start_idx:{start_idx}, end_idx:{end_idx}'''
            # for idx in pred_data[vid]:
                # assert idx <= end_idx and start_idx <= idx, f'''{vid} pred_idx:{idx}, start_idx:{start_idx}, end_idx:{end_idx}'''


if __name__ == '__main__':
    main()


import json
import copy
input_gold = "_version3_roop_0_seed_0_selectalpha_0.5_caption_finetuning_sec_greedy_gold.json"
output_gold = "version3_roop_0_seed_0_selectalpha_0.5_caption_finetuning_sec_greedy_gold.json"
gold_data = json.load(open(input_gold))
output_data = copy.deepcopy(gold_data)
for vid in output_data.keys():
    for i in range(len(output_data[vid])):
        start_idx = output_data[vid][i]["start"]
        end_idx = output_data[vid][i]["end"]
        ann_secs = []
        for ann_idx in output_data[vid][i]["ann_secs"]:
            if ann_idx <= end_idx and start_idx <= ann_idx:
                ann_secs.append(ann_idx)
        output_data[vid][i]["ann_secs"] = ann_secs

json.dump(output_data, open(output_gold, 'w'), indent=2)

# print(json.dumps(gold_data, indent=2), file=open("version3_roop_0_seed_0_selectalpha_0.5_caption_finetuning_sec_greedy_gold.json", 'w'))
        # for idx in pred_data[vid]: