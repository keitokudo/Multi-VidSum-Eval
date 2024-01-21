"""
実験結果をエクセルの表形式にまとめてくれるコード
"""

import argparse
import logzero
from logzero import logger
import logging
from os import path
from typing import List
import numpy as np
import pickle


from eval_frame_match import output_result
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input file dir')
    parser.add_argument("--prefix", type=str, default="validation_testing_suzuki_subword_eval_frame_match_centerized", help="ファイル名の名前")
    parser.add_argument("--suffix", type=str, default="using_val1val2", help="ファイル名の名前")

    args = parser.parse_args()
    return args

def calc_standard_error(x):
    """標準誤差を計算"""
    return np.std(x, ddof=1) / np.sqrt(len(x))


def main():
    args = parse_args()
    # logzero.logfile(args.log)  # 追加: logfileの作成
    logger.info(args)
    print("Condition", "Selector", "Architecture", "AKM", "AKM(cos)", "B@1", "B@2", "B@3", "B@4", sep='\t')

    for search_way in ["greedy", "dp"]:
        for condition in ["oracle", "densecap", "random"]:
            # seed_list = range(5) if condition == "random" else
            seed_list =  range(1,2)
            ret = defaultdict(list)
            # mode = "" if condition == "oracle" else f"_{search_way}_{condition}"
            if condition == 'oracle':
                mode = "_densecap"
            elif condition == "densecap":
                mode = f"_{search_way}_{condition}"
            elif condition == "random":
                mode = f"_{search_way}_random_sampling_densecap"
                seed_list=[0]
            else:
                raise ValueError
            for seed in seed_list:
                df = pickle.load(open(f"{args.input}/eval/{args.prefix}{mode}_sub_movie_resnet_{seed}_{args.suffix}.pickle",'rb'))

                output = output_result(df)
                for k,v in output.items():
                    ret[k].append(v)
                blue_file = f"{args.input}/eval/{args.prefix}{mode}_sub_movie_resnet_{seed}_{args.suffix}/result.txt"
                with open(blue_file) as f:
                    line = f.readline().rstrip()
                    b1, b2, b3, b4 = map(float,line.split()[3].split('/'))
                    ret["b1"].append(b1)
                    ret["b2"].append(b2)
                    ret["b3"].append(b3)
                    ret["b4"].append(b4)
                
            print(search_way, condition, sep='\t', end='\t')
            for term in ["acc_per_cap", "sim_value"]:
                print(f"{np.mean(ret[term]):.3f}", end='')
                if 1 < len(ret[term]):
                    print(f"±{calc_standard_error(ret[term]):.3f}", end='')
                print('\t', end='')
            # BLUE表示
            for term in ["b1", "b2", "b3", "b4"]:
                print(f"{np.mean(ret[term]):.3f}", end='')
                if 1 < len(ret[term]):
                    print(f"±{calc_standard_error(ret[term]):.3f}", end='')
                print('\t', end='')
            print()

if __name__ == '__main__':
    main()
