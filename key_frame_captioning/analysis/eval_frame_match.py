import argparse
from os import path
import numpy as np
import pickle
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=path.abspath, help='input file path')
    args = parser.parse_args()
    return args


def output_result(df):
    output_json = dict()
    output_json["acc_per_movie_cnt"] = sum(df["p_comb"] == df["g_comb"])
    output_json["movie_cnt"] = len(df)
    output_json["acc_per_cap_cnt"] = sum(df[["p_comb", "g_comb"]].apply(
        lambda x: sum(a == b for a, b in zip(x["p_comb"], x["g_comb"])), axis=1))
    output_json["cap_cnt"] = sum(df["p_comb"].apply(len))
    output_json["sim_value"] = np.mean(df["sim_value"])

    output_json["acc_per_movie"] = output_json["acc_per_movie_cnt"] / \
        output_json["movie_cnt"]
    output_json["acc_per_cap"] = output_json["acc_per_cap_cnt"] / output_json["cap_cnt"]

    return output_json


def main():
    args = parse_args()
    df = pickle.load(open(args.input, 'rb'))
    print(json.dumps(output_result(df),indent=2))


if __name__ == '__main__':
    main()
