"""selectしたsegmentの良さを評価する"""


import argparse
from collections import defaultdict
import logzero
from logzero import logger
import logging
from os import path
from typing import List
import numpy as np
import json



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-it', '--input_tsv', type=path.abspath, help='input file path', default="/work01/video-and-nlp-2021/yahoo/data/sentenceanno_result_allin.sort_sid.trimed.tsv")        
    parser.add_argument(
        '-ij', '--input_json', type=path.abspath, help='input json file path')
    # parser.add_argument(
    #     '-o', '--output', type=path.abspath, help='output file path')
#    parser.add_argument('--log',
#                        default="log.txt", help='Path to log file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)

    gold_segment = defaultdict(list)
    with open(args.input_tsv, 'r') as f:
        prev_caption = None
        prev_start_sec = None
        prev_end_sec = None
        for line in f:
            line = line.rstrip('\r\n')
            _, vid, _, _,  ann_sec, caption, _, start_sec, end_sec,  mode, v_sec = line.split(
                '\t')
            start_sec, end_sec = float(start_sec), float(end_sec)
            ann_sec = float(ann_sec)

            #captionが同じでsegmentが異なるというcaptionの対応するために修正
            if not (caption == prev_caption and start_sec == prev_start_sec and end_sec == prev_end_sec):
                # 新しいsegmentが始まる
                gold_segment[vid].append([])
            gold_segment[vid][-1].append(ann_sec)

            prev_caption = caption
            prev_start_sec = start_sec
            prev_end_sec = end_sec

    select_data = json.load(open(args.input_json))

    vid_set = set(gold_segment.keys()) & set(select_data.keys())            
    
    score_list = []
    for vid in vid_set:
        for i in range(len(select_data[vid])):
            start_sec, end_sec = select_data[vid][i]["timestamp"]
            max_score = max([eval(start_sec, end_sec, ann_secs) for ann_secs in gold_segment[vid]])
            score_list.append(max_score)
    print(np.mean(score_list))

def eval(start_sec, end_sec, ann_secs):
    return (sum([start_sec <= ann_sec and ann_sec <= end_sec for ann_sec in ann_secs])/len(ann_secs)) / (end_sec - start_sec)



if __name__ == '__main__':
    main()
