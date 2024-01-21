import argparse
from logzero import logger
from os import path
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', type=path.abspath, help='output directory path')
    parser.add_argument(
        '-p', "--pickle_file", type=path.abspath, help='evaluate pickle file')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    logger.info(args)
    os.makedirs(args.output, exist_ok=True)
    pickle_df = pickle.load(open(args.pickle_file, 'rb'))

    vid_list = pickle_df["vid"].apply(lambda x:f"v_{x}")

    with open(f"{args.output}/gold.txt", 'w') as f_gold, open(f"{args.output}/pred.txt", 'w') as f_pred:
        for vid in vid_list:
            gold_captions = pickle_df[pickle_df["vid"]==vid[2:]].iloc[0]["g_comb_captions"]
            pred_captions = pickle_df[pickle_df["vid"]==vid[2:]].iloc[0]["p_comb_captions"]
            assert len(gold_captions) == len(pred_captions)
            for i in range(len(gold_captions)):
                print(gold_captions[i].strip(), file=f_gold)
                print(pred_captions[i].strip(), file=f_pred)


if __name__ == '__main__':
    main()
