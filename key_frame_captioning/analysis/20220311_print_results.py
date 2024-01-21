import pickle
from glob import glob

import os
import sys
from eval_frame_match import output_result

root_path="/work01/video-and-nlp-2021/yahoo/results/20211220_bilstm_sub_movie/eval"
search_mode="greedy"
seed=0
alpha_list = [0, 0.1, 0.2,0.3, 0.4,0.5,0.6,0.7,0.8, 0.9, 1.0]
caption_model="finetuning"
intial_cap="sec"



for alpha in alpha_list:
    for roop in  range(5):
        for output_name in glob(f'''{root_path}/version3_roop_{roop}_seed_{seed}_selectalpha_{alpha}_caption_{caption_model}_{intial_cap}_{search_mode}/'''):
            df = pickle.load(open(f"{output_name[:-1]}.pickle",'rb'))
            results = output_result(df)
            AKM_ex = results["acc_per_cap"]
            AKM_cos = results["sim_value"]

            with open(f'''{output_name}result.txt''') as f:
                line = f.readline().rstrip()
                b1, b2, b3, b4 = map(float,line.split()[3].split('/'))
            
            print(caption_model, search_mode, alpha, roop, AKM_ex, AKM_cos, b1, b2,b3,b4,sep='\t')
    print()