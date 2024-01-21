import click
import yaml
import os
import pickle
import json
from pathlib import Path
from collections import defaultdict
import copy
import random
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# logger
import logzero
from logzero import logger
import logging


import numpy as np
import pandas as pd

# torch
import torch
from torch.utils.data import DataLoader

# our implements
from Dataset import ANetDataset
from tokenizer import get_tokenizer
from models.baseline import BaselineModel
from analysis.eval_frame_match import output_result
from key_frame_captioning.eval.eval_frame_match_multiple import eval_frame_match_multiple


@click.group()
def cli():
    pass


def set_log_level(debug):
    '''
    ログのレベルを指定する
    '''
    if debug is True:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    return


def create_golds_captions(dataloader):
    golds = defaultdict(list)
    gold_captions = defaultdict(list)
    gold_segment_timestamps = defaultdict(list)

    with torch.no_grad():
        for b_resnet, b_bni, b_sent_ids, _, b_frame_masks, b_sent_mask, b_vid, _, (b_original_gold_labels, b_start_idx, _, b_original_frame_cnt), (_, b_captions), b_gold_captions, _, (b_start_sec, b_end_sec) in dataloader:
            for i in range(b_resnet.size(0)):
                golds[b_vid[i]].append(
                    b_original_gold_labels[i][:b_original_frame_cnt[i]].detach().cpu().numpy().tolist())
                gold_captions[b_vid[i]].append(b_gold_captions[i])
                gold_segment_timestamps[b_vid[i]].append([b_start_sec[i], b_end_sec[i]])
                
    return golds, gold_captions, gold_segment_timestamps

def eval_frame_match_version3(pred, cap, output_dir, reference_dataloader, max_captions, output_file_name, feature, method, is_random=False, use_sub_movie=False, alpha=1.0, is_predict_frame=True, selected_alpha=0.5, tokenizer=None, eval_config=None, preprocessed_caption_dir=None):

    test_golds, test_gold_captions, _ = create_golds_captions(reference_dataloader)

    # devとtestのgoldsをまとめる
    golds = defaultdict(list)
    gold_captions = defaultdict(list)
    test_vids_set = set(test_golds.keys())
    

    ### predデータのロード
    
    ## キャプションのロード
    # 高速化用
    ## キーフレームのロード
    if cap == 'self':
        dj = json.load(open(f'{pred}', 'r'))
        preds = {}
        #scores = {}
        all_captions = defaultdict(dict)
        for d in dj['result']:
            preds[d['video_id']] = d['hyp_frame_idxs']
            for i, t in enumerate(d['hyp_texts']):
                all_captions[d['video_id']][int(d['hyp_frame_idxs'][i])] = t
            #scores[d['video_id']] = d['score'][0]
    else:
        all_captions = pickle.load(
            (Path(preprocessed_caption_dir) / f"all_captions_{cap}.pkl").open(
                mode="rb"
            )
        )
        dj = json.load(open(f'{pred}', 'r'))
        preds = {}
        #scores = {}
        for d in dj['result']:
            preds[d['video_id']] = d['n_best_frame_index'][0]
            #scores[d['video_id']] = d['score'][0]
            
            
    ### predとgoldのマッチング
    pred_vids_set = set(preds.keys())

    # vid_set = pred_vids_set & (dev_vids_set | test_vids_set)
    vid_set = pred_vids_set
    
    for vid in test_vids_set:
        golds[vid].extend(test_golds[vid])
        gold_captions[vid].extend(test_gold_captions[vid])
    
    output_file = f"{output_dir}/{output_file_name}.pickle"
    output_df = defaultdict(list)
    image_set = 'fixed_eval'
    test_pred_captions = dict()
    
    for vid in vid_set:
        pred_index = np.array(preds[vid])
        frame_data_all = np.array(golds[vid])
            
        test_pred_captions[vid] = [all_captions[vid][frame] for frame in pred_index]
        
        if feature == "resnet":
            vis_feat = np.load(
                f"{eval_config['frame_match_feature_root']}/{image_set}/{vid}_resnet.npy"
            )
        elif feature == "bni":
            vis_feat = np.load(
                f"{eval_config['frame_match_feature_root']}/{image_set}/{vid}_bn.npy"
            )
        elif feature == "clip":
            vis_feat = np.load(
                f"{eval_config['frame_match_feature_root']}/{image_set}/{vid}_clip.npy"
            )
        else:
            raise ValueError
        
        pred_index = [min(vis_feat.shape[0]-1, p) for p in pred_index]

        logger.debug("======= start =========")
        logger.debug(vid)
        logger.debug(f"pred_index {pred_index}")
        logger.debug(f"captions {test_pred_captions[vid]}")
        logger.debug(f"gold index {list(zip(*np.where(frame_data_all == 1)))}")
        logger.debug(vis_feat.shape)

        assert vis_feat.shape[0] == frame_data_all.shape[1], f"{vid}: {vis_feat.shape[0]}, {frame_data_all.shape[1]}"

        p_comb, g_comb, sim_value, selected_pred_captions, selected_gold_captions = eval_frame_match_multiple(pred_index, vis_feat, frame_data_all, test_pred_captions[vid], gold_captions[vid], alpha)
        selected_pred_index = copy.deepcopy(p_comb)
            
        if not (g_comb is not None and p_comb is not None and selected_pred_captions is not None and selected_pred_index is not None):
            raise ValueError(
                f"{vid} caused error. we can't select invalid frame g_comb:{g_comb}, p_comb:{p_comb} selectes_p_cap:{selected_pred_captions} selected_p_idx:{selected_pred_index}"
            )
        

        # gold_frame_idx_list = list(zip(*np.where(frame_data_all == 1)))

        logger.debug(f"sim_value: {sim_value}")

        assert len(g_comb) == len(p_comb), f"{len(g_comb)}, {len(p_comb)}"
        assert len(p_comb) == len(selected_pred_captions), f"{len(p_comb)}, {len(selected_pred_captions)}"
        assert len(selected_pred_index) == len(g_comb), f"{len(selected_pred_index)}, {len(g_comb)}"

        tmp = list(zip(g_comb, p_comb, selected_pred_captions, selected_gold_captions))
        tmp = sorted(tmp, key=lambda x:x[0])
        g_comb, p_comb, p_comb_captions, g_comb_captions = list(zip(*tmp))

        logger.debug(f"p_comb {p_comb}")
        logger.debug(f"g_comb {g_comb}")
        # logger.debug(f"g_comb_captions {g_comb_captions}")
        logger.debug("======== finish ===========")

        #g_combのindexが早い順番にp_combとg_comb_captionをsort
        output_df["vid"].append(vid)
        output_df["pred"].append(selected_pred_index)
        output_df["p_comb"].append(p_comb)
        output_df["g_comb"].append(g_comb)
        output_df["sim_value"].append(sim_value)
        output_df["p_comb_captions"].append(p_comb_captions)
        output_df["g_comb_captions"].append(g_comb_captions)


    df = pd.DataFrame()
    for key, value in output_df.items():
        df[key] = value
    logger.info(output_result(df))
    pickle.dump(df, open(output_file, 'wb'))


@cli.command()
@click.option("--pred", type=click.Path(exists=True))
@click.option("--cap", type=str)
@click.option('--model_config_file', type=click.Path(exists=True), help="modelのパラメータなどの設定")
@click.option('--eval_config_file', type=click.Path(exists=True), help="config file")
@click.option('--res_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
@click.option('--output_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
@click.option("--debug", type=bool, is_flag=True, default=False, help="デバッグモードにする")
@click.option('--gpu', type=bool, is_flag=True, default=False, help="gpuを使うかどうか")
@click.option("--num_workers", type=int, default=6, help="前処理のcpu使用数")
@click.option("--max_captions", type=int, default=100000000, help="評価に使用する動画の最大キャプション数")
@click.option("--output_file_name", type=str, default="eval_frame_match_using_val1val2", help="出力ファイル名")
@click.option("--random_frame_sampling", type=bool, is_flag=True, default=False, help="フレーム抽出をランダムに行うベースラインの評価を行う")
@click.option("--feature",  type=click.Choice(['resnet', 'bni'], case_sensitive=False), default="resnet", help="フレームの比較に使用する特徴量")
@click.option("--use_sub_movie", type=bool, is_flag=True, default=False, help="動画を該当箇所のみ切り抜いて使用する")
@click.option('--dense_cap_file', type=click.Path(exists=True), default=None, help="densecapの出力jsonファイル")
@click.option("--seed", type=int, default=1)
@click.option('--use_only_movie_exist_vid', type=bool, is_flag=True, default=False, help="動画ファイルに含まれるもののみを使うかどうか")
@click.option("--alpha", type=float, default=1.0, help="コサイン類似度とbleuスコアとの比率")
@click.option("--is_predict_frame", type=bool, is_flag=True, default=False, help="モデルを用いて予測を行う")
@click.option("--method",  type=click.Choice(['greedy', 'dp', 'random'], case_sensitive=False), help="評価に使用するセグメントを選ぶ手法")
@click.option("--selected_alpha", type=float, help="dpやgreedyにおいてframe scoreとcaption scoreの割合")
@click.option("--preprocessed_caption_dir", type=click.Path(exists=True), help="前処理済みのキャプションファイルがあるディレクトリ", required=True)
def evaluate_version3(pred, cap, model_config_file, eval_config_file, res_dir, output_dir, debug, gpu, num_workers, max_captions, output_file_name, random_frame_sampling, feature, use_sub_movie, dense_cap_file, seed, use_only_movie_exist_vid, alpha, is_predict_frame, method, selected_alpha, preprocessed_caption_dir):
    '''
    version3用の評価スクリプトを再利用する（データのロードとかめんどくさい）
    modelとか必要ないけど動かすのに必要
    '''
    #image_set = "validation_testing_version3"
    device = "cuda" if torch.cuda.is_available() and gpu is True else "cpu"
    logger.info(f"use {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    random.seed(seed)


    eval_config = yaml.safe_load(open(eval_config_file))
    model_config = yaml.safe_load(open(model_config_file))

    set_log_level(debug)


    logger.info(eval_config)


    tokenizer, vocab, pad_token_id = get_tokenizer(
        model_config["model"]["language_model"], eval_config["dataset_file"])

    model = BaselineModel(model_config["model"], device)

    # モデルをloadx
    model.load_state_dict(torch.load(f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_{eval_config['use_model']}.pt")["model"])
    model.eval()
    
    reference_dataset = ANetDataset(
        eval_config,
        tokenizer,
        pad_token_id,
        vocab,
        debug,
        image_set="fixed_eval",
        use_sub_movie=use_sub_movie,
        use_only_movie_exist_vid=use_only_movie_exist_vid
    )
    reference_dataloader = DataLoader(
        reference_dataset,
        eval_config['batchsize'],
        num_workers=num_workers,
        shuffle=False,
        collate_fn=reference_dataset.collate_fn
    )    
    eval_frame_match_version3(pred, cap, output_dir, reference_dataloader, max_captions, output_file_name, feature, method, is_random=random_frame_sampling, use_sub_movie=use_sub_movie,alpha=alpha, is_predict_frame=is_predict_frame, selected_alpha=selected_alpha, tokenizer=tokenizer, eval_config=eval_config, preprocessed_caption_dir=preprocessed_caption_dir)
    

if __name__ == "__main__":
    cli()
