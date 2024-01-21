import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import  Sigmoid

from logzero import logger
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle
import json

from itertools import chain
import copy

from eval.eval_frame_match import eval_func
from analysis.eval_frame_match import output_result
# from FrameDetectDataset import FrameANetDataset

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from create_sub_movie_fetures import CreateSubMovie
from preprocess.video_preprocess import VideoConverter
from preprocess.fairseq_preprocess.select_n_segment import  greedy, dp, random_sampling


from key_frame_captioning.eval.eval_frame_match_multiple import eval_frame_match_multiple

from preprocess.sec_preprocess import Preprocess


class Evaluator:
    def __init__(self, model, device, root_dir, eval_config, vocab, image_set, debug=False, criterion=None, frame_model=None):
        self.device = device
        self.root_dir = root_dir
        self.model = model.to(device)
        self.model.eval()
        self.eval_config = eval_config
        self.vocab = vocab
        self.debug = debug
        self.image_set = image_set
        self.i2w = dict([(v, k) for k, v in self.vocab.items()])
        self.criterion = criterion
        if frame_model is not None:
            self.frame_model = frame_model.to(device)
            self.frame_model.eval()
        # self.csm = CreateSubMovie()
        self.vc = VideoConverter(device)
        self.pp = Preprocess()


    def eval_frame_match_with_two_model(self, dataloader, max_captions, output_file_name, feature, is_random=False, use_sub_movie=False):
        
            
        sampling_preds = defaultdict(list)
        # sampling_golds = defaultdict(list)
        sent_ids = defaultdict(list)
        sampling_frame_cnt = dict()
        ann_secs_dict = defaultdict(list)

        start_time = time.time()

        with torch.no_grad():
            for b_resnet, b_bni, b_sent_ids, _, b_frame_masks, b_sent_mask, b_vid, _, (b_original_gold_labels, b_start_idx, _, b_original_frame_cnt), (b_ann_secs), *_ in dataloader:

                b_resnet = b_resnet.to(self.device)
                b_bni = b_bni.to(self.device)
                b_sent_ids = b_sent_ids.to(self.device)
                b_original_gold_labels = b_original_gold_labels.to(self.device)
                b_sent_mask = b_sent_mask.to(self.device)
                b_frame_masks = b_frame_masks.to(self.device)

                b_sent_cnt = b_sent_mask.sum(-1)

                if is_random is True:
                    index = [torch.randint(b_frame_masks[i].sum(), (1,)).squeeze(0).item() for i in range(len(b_vid))]
                else:
                    outputs = self.model(b_resnet, b_bni, b_sent_ids,
                                        b_frame_masks, b_sent_mask)
                    _, index = torch.max(outputs, dim=-1)
                    index = index.view(-1).detach().cpu().numpy().tolist()

                for i in range(b_resnet.size(0)):
                    if use_sub_movie is True:
                        sampling_preds[b_vid[i]].append(index[i] + b_start_idx[i])
                    else:
                        sampling_preds[b_vid[i]].append(index[i])

                    # sampling_golds[b_vid[i]].append(
                    #     b_original_gold_labels[i][:b_original_frame_cnt[i]].detach().cpu().numpy().tolist())

                    sent_ids[b_vid[i]].append(b_sent_ids[i][:b_sent_cnt[i]])
                    sampling_frame_cnt[b_vid[i]] = b_original_frame_cnt[i]
                    ann_secs_dict[b_vid[i]].append(b_ann_secs[i])

        # print(f"segment pred finished. {time.time()-start_time:.2f} sec")

        output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"
        output_df = defaultdict(list)

        if feature == 'resnet':
            feature_dim = b_resnet.size(-1)
            numpy_mode_name = "resnet"
        elif feature == 'bni':
            feature_dim = b_bni.size(-1)
            numpy_mode_name = "bn"
        else:
            raise ValueError(f"{feature} is unknown")

        for vid in sampling_preds.keys():
            movie_file = f'{self.eval_config["movie_root"]}/v_{vid}.mp4'
            # 動画情報を取得
            movie_info = self.vc.get_movie_info(movie_file)
            if movie_info is None:
                logger.warning(f"{vid} is not exist in  mp4 file")
                continue
            sampling_pred_index_list = sampling_preds[vid]
            

            if max_captions < len(sampling_pred_index_list):
                logger.info(f"{vid} was skipped because it has {len(sampling_pred_index_list)} captions")
                continue
            
            # if feature == "resnet":
            #     vis_feat = np.load(
            #         f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_resnet.npy")
            # elif feature == "bni":
            #     vis_feat = np.load(
            #         f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_bn.npy")
            # else:
            #     raise ValueError
            #    予測したindexとgoldのindex部分だけ格納する用のfeaturesを作成
            vis_feat = np.zeros((movie_info["frame_count"], feature_dim))
            


        #     logger.debug(vid)
        #     logger.debug(sampling_pred_index_list)
        #     logger.debug(list(zip(*np.where(frame_data_all == 1))))
        #     logger.debug(vis_feat.shape)

    #     ## TO DO#######
    #     # eval_funcを動画全体入れなくても動くように実装（predとgold　indexだけ持ってれば大丈夫だと思う）
    #     # sampling_sec秒でのindexからstart_sec とend_secを計算する
    #     # その時間分の特徴量を取得する

            frame_pred_start_time = time.time()

            start_end_sec_list = []
            for p_i in sampling_pred_index_list:
                # predの予測indexからstart secとend secを取得
                start_sec, end_sec = self.vc.get_start_end_sec_from_index(p_i, movie_info["total_sec"], sampling_frame_cnt[vid])
                logger.debug(f"{p_i}-> start sec: {start_sec:.2f}, end sec: {end_sec:.2f}")
                start_end_sec_list.append([start_sec, end_sec])


            # start_secとend_secのリストから特徴量のリストを作成（indexが対応している）
            b_bni, b_resnet = self.vc.get_submovie_tensor_from_start_to_end_sec(movie_file, start_end_sec_list)
            b_bni = pad_sequence(b_bni, batch_first=True, padding_value=0).to(self.device)
            b_resnet = pad_sequence(b_resnet, batch_first=True, padding_value=0).to(self.device)
            b_frame_masks = torch.ones_like(b_bni[:, :, 0]).long().to(self.device)
            b_frame_masks[b_bni.sum(-1)== 0] = 0
            b_frame_cnt = b_frame_masks.sum(-1)
            
    #         他のデータを整形する
            pad_id = 0 #0でpaddingを決め打ちしちゃってる
            b_sent_ids = pad_sequence(sent_ids[vid], batch_first=True, padding_value=pad_id).to(self.device)
            # b_frame_masks = torch.ones_like(b_bni.size()[:-1])
            b_sent_mask = (b_sent_ids != pad_id).long().to(self.device)


            # 秒数0.5秒の中でのindexを予測する
            if is_random is True:
                sub_frame_sampling_pred_index_list = [torch.randint(b_frame_masks[i].sum(), (1,)).squeeze(0).item() for i in range(len(b_bni.size(0)))]
            else:
                outputs = self.model(b_resnet, b_bni, b_sent_ids,
                                    b_frame_masks, b_sent_mask)
                _, sub_frame_sampling_pred_index_list = torch.max(outputs, dim=-1)
                sub_frame_sampling_pred_index_list = sub_frame_sampling_pred_index_list.view(-1).detach().cpu().numpy().tolist()

            # print(f"index pred finished. {time.time() - frame_pred_start_time:.2f}sec")

            # indexから秒数を取得する
            sub_frame_pred_sec_list = [self.vc.get_start_end_sec_from_index(sub_frame_sampling_pred_index_list[i], start_end_sec_list[i][1] - start_end_sec_list[i][0], b_frame_cnt[i]) for i in range(len(start_end_sec_list))]
            # start secとend　secの間を秒数とする
            sub_frame_pred_sec_list = list(map(lambda x:sum(x)/2, sub_frame_pred_sec_list))

            assert len(sub_frame_pred_sec_list) == len(start_end_sec_list), f"{len(sub_frame_pred_sec_list)}, {len(start_end_sec_list)}" 

            overall_pred_sec_list = [start_end_sec[0] + sub_frame_pred_sec for start_end_sec, sub_frame_pred_sec in zip(start_end_sec_list, sub_frame_pred_sec_list)]
            # indexに直す　
            overall_pred_index_list = [self.vc.get_index_from_sec(op_sec, movie_info["total_sec"], movie_info["frame_count"]) for op_sec in overall_pred_sec_list]

            logger.debug(f"""
                sampling pred :{sampling_pred_index_list[0]}/{sampling_frame_cnt[vid]} ({start_end_sec_list[0][0]:.2f}~{start_end_sec_list[0][1]:.2f}sec)
                sub_frame_pred: {sub_frame_sampling_pred_index_list[0]}/{b_frame_cnt[0]} ({sub_frame_pred_sec_list[0]:.2f}sec)
                overall_pred: {overall_pred_index_list[0]}/{movie_info['frame_count']} ({overall_pred_sec_list[0]:.2f}sec)
            """)



    #         予測したindexを元にその特徴量を取得しvisfeatに格納
            assert len(overall_pred_index_list) == len(sub_frame_sampling_pred_index_list), f"{len(overall_pred_index_list)}, {len(sub_frame_sampling_pred_index_list)}"
            b_resnet = b_resnet.to("cpu").detach().numpy().copy()
            b_bni = b_bni.to("cpu").detach().numpy().copy()
            for i, (opi, sfspi) in enumerate(zip(overall_pred_index_list, sub_frame_sampling_pred_index_list)):
                if feature == 'resnet':
                    vis_feat[opi] = b_resnet[i][sfspi]
                elif feature == 'bni':
                    vis_feat[opi] = b_bni[i][sfspi]
                else:
                    raise ValueError(f"{feature} is unknown")


            # print(f"overall pred index finished. {time.time() - frame_pred_start_time:.2f}sec")

            gold_index_list = [[self.vc.get_index_from_sec(sec, movie_info['total_sec'], movie_info['frame_count']) for sec in sec_list] for sec_list in ann_secs_dict[vid]]
            # serialized_gold_index_list = list(chain.from_iterable(gold_index_list))
            # g_bni, g_resnet = self.vc.get_submovie_tensor_from_sec(movie_file, serialized_gold_index_list)
            # g_bni = g_bni.to("cpu").detach().numpy().copy()
            # g_resnet = g_resnet.to("cpu").detach().numpy().copy()
            # for i, sgi in enumerate(serialized_gold_index_list):
            #     if feature == 'resnet':
            #         vis_feat[sgi] = g_resnet[i]
            #     elif feature == 'bni':
            #         vis_feat[sgi] = g_bni[i]
            #     else:
            #         raise ValueError(f"{feature} is unknown")

            # あらかじめ作ってあるものを使った方が早い
            import glob
            assert len(gold_index_list) == len(ann_secs_dict[vid]), f"{len(gold_index_list)}, {len(ann_secs_dict[vid])}"
            for i, sec_list in enumerate(ann_secs_dict[vid]):
                assert len(gold_index_list[i]) == len(sec_list), f"{len(gold_index_list[i])}, {len(sec_list)}"
                for j, sec in enumerate(sec_list):
                    query = f"{self.eval_config['frame_match_feature_root']}/validation/v_{vid}*_{sec}_{numpy_mode_name}.npy"
                    numpy_path = list(glob.glob(query))
                    assert len(numpy_path) == 1, query
                    *_, start_sec, end_sec, _, _ = bni_path.split('_')
                    start_sec, end_sec = float(start_sec), float(end_sec)
                    features = np.load(numpy_path)
                    index = self.vc.get_index_from_sec(sec, end_sec - start_sec, frame_cnt)
                    vis_feat[gold_index_list[i][j]] = features[index].copy()
                    


    #         gold_indexを元にしてvis_featに特徴量を入れる

            # print(f"index gold finished. {time.time() - frame_pred_start_time:.2f}sec")

    #         各セグメントの frame_data_allを作成
            frame_data_all = np.zeros((len(sampling_pred_index_list), movie_info["frame_count"]))
            for i in range(len(sampling_pred_index_list)):
                for gi in gold_index_list[i]:
                    frame_data_all[i][gi] = 1
            
            # print(f"frame data all finished. {time.time() - frame_pred_start_time:.2f}sec")

    # #         for i in range(b_resnet.size(0)):
    # #             if use_sub_movie is True:
    # #                 preds[vid][i] = self.create_index(preds[vid][i], index[i] + b_start_idx[i], self.eval_config["sampling_sec"], movie_info["frame_count"])
    # #             else:
    # #                 preds[vid][i] = self.create_index(preds[vid][i], index[i], self.eval_config["sampling_sec"], movie_info["frame_count"])

    # #         for index in range(np.where(golds[vid].detach().numpy() == 1)[1])
    # #             gold_index = self.create_index(index, , self.eval_config["sampling_sec"], movie_info["frame_count"])

    # #         frame_data_all = self.create_frame_data_all(golds[vid], b_gold_label)        
                

    # #         pred_index = self.createsampling_pred_index_list(sampling_pred_index_list, )
    # #     #frame modelでその区間でのindexを作成する
    # #     #全体の動画での時間を算出する


            p_comb, g_comb, sim_value = eval_func(
                overall_pred_index_list, vis_feat, frame_data_all)
            p_comb = p_comb.tolist()
            g_comb = g_comb.tolist()

            # print(f"sim value. {time.time() - frame_pred_start_time:.2f}sec")

            output_df["vid"].append(vid)
            output_df["pred"].append(overall_pred_index_list)
            output_df["p_comb"].append(p_comb)
            output_df["g_comb"].append(g_comb)
            output_df["sim_value"].append(sim_value)

        df = pd.DataFrame()
        for key, value in output_df.items():
            df[key] = value
        logger.info(output_result(df))
        pickle.dump(df, open(output_file,'wb'))


        # Convert to 


        

    def create_golds_captions(self, dataloader):
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

    def create_preds_golds_captions(self, dataloader, is_random=False, use_sub_movie=False, is_predict_frame=False):
        preds = defaultdict(list)
        golds = defaultdict(list)
        pred_captions = defaultdict(list)
        gold_captions = defaultdict(list)
        gold_timestamps = defaultdict(list)
        pred_frame_scores = defaultdict(list)
        pred_caption_scores = defaultdict(list)

        sigmoid = Sigmoid()

        with torch.no_grad():
            if is_predict_frame is True:
                for b_resnet, b_bni, b_sent_ids, _, b_frame_masks, b_sent_mask, b_vid, _, (b_original_gold_labels, b_start_idx, _, b_original_frame_cnt), (_, b_captions), b_gold_captions, (_, _, b_caption_scores), (b_start_sec, b_end_sec) in dataloader:
                    b_resnet = b_resnet.to(self.device)
                    b_bni = b_bni.to(self.device)
                    b_sent_ids = b_sent_ids.to(self.device)
                    b_original_gold_labels = b_original_gold_labels.to(self.device)
                    b_sent_mask = b_sent_mask.to(self.device)
                    b_frame_masks = b_frame_masks.to(self.device)

                    if is_random is True:
                        index = [torch.randint(b_frame_masks[i].sum(), (1,)).squeeze(0).item() for i in range(len(b_vid))]
                        frame_score=[0 for _ in range(len(b_vid))]
                    else:
                        outputs = self.model(b_resnet, b_bni, b_sent_ids,
                                            b_frame_masks, b_sent_mask)
                        frame_score, index = torch.max(outputs, dim=-1)
                        index = index.view(-1).detach().cpu().numpy().tolist()
                        frame_score = ((sigmoid(frame_score) + 1)/2).view(-1).detach().cpu().numpy().tolist()

                    for i in range(b_resnet.size(0)):
                        if use_sub_movie is True:
                            preds[b_vid[i]].append(index[i] + b_start_idx[i])
                        else:
                            preds[b_vid[i]].append(index[i])

                        golds[b_vid[i]].append(
                            b_original_gold_labels[i][:b_original_frame_cnt[i]].detach().cpu().numpy().tolist())
                        
                        pred_captions[b_vid[i]].append(b_captions[i])
                        gold_captions[b_vid[i]].append(b_gold_captions[i])
                        gold_timestamps[b_vid[i]].append([b_start_sec[i], b_end_sec[i]])
                        pred_frame_scores[b_vid[i]].append(frame_score[i])
                        pred_caption_scores[b_vid[i]].append(b_caption_scores[i])

                assert len(preds) == len(pred_frame_scores), f'''{len(preds)}, {len(pred_frame_scores)}'''
                assert len(preds) == len(gold_timestamps), f'''{len(preds)}, {len(gold_timestamps)}'''
            else:
                for b_resnet, b_bni, b_sent_ids, _, b_frame_masks, b_sent_mask, b_vid, _, (b_original_gold_labels, b_start_idx, _, b_original_frame_cnt), (_, b_captions), b_gold_captions, (frame_idx, frame_score, b_caption_scores), (b_start_sec, b_end_sec) in dataloader: 
                    if is_random is True:
                        # assert False, "it still be not implement"
                        index = [
                            torch.randint(b_frame_masks[i].sum(), (1,)).squeeze(0).item()
                            for i in range(len(b_vid))
                        ]
                        for i in range(b_resnet.size(0)):
                            if use_sub_movie is True:
                                preds[b_vid[i]].append(index[i] + b_start_idx[i])
                            else:
                                preds[b_vid[i]].append(index[i])
                                
                            golds[b_vid[i]].append(
                                b_original_gold_labels[i][:b_original_frame_cnt[i]].detach().cpu().numpy().tolist()
                            )
                            pred_captions[b_vid[i]].append(b_captions[i])
                            gold_captions[b_vid[i]].append(b_gold_captions[i])
                            gold_timestamps[b_vid[i]].append([b_start_sec[i], b_end_sec[i]])
                            pred_frame_scores[b_vid[i]].append(frame_score[i])
                            pred_caption_scores[b_vid[i]].append(0)
                                
                    else:
                        index = frame_idx
                        for i in range(b_resnet.size(0)):
                            preds[b_vid[i]].append(index[i])
                     
                            golds[b_vid[i]].append(
                                b_original_gold_labels[i][:b_original_frame_cnt[i]].detach().cpu().numpy().tolist())
                            
                            pred_captions[b_vid[i]].append(b_captions[i])
                            gold_captions[b_vid[i]].append(b_gold_captions[i])
                            gold_timestamps[b_vid[i]].append([b_start_sec[i], b_end_sec[i]])
                            pred_frame_scores[b_vid[i]].append(frame_score[i])
                            pred_caption_scores[b_vid[i]].append(b_caption_scores[i])

        return preds, golds, pred_captions, gold_captions, gold_timestamps, pred_frame_scores, pred_caption_scores

    def eval_frame_match(self, dataloader, max_captions, output_file_name, feature, is_random=False, use_sub_movie=False):
        preds, golds, pred_captions, gold_captions = self.create_preds_golds_captions(dataloader, is_random, use_sub_movie)

        output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"
        output_df = defaultdict(list)
        for vid in preds.keys():
            pred_index = preds[vid]
            frame_data_all = np.array(golds[vid])

            if max_captions < len(pred_index):
                logger.info(f"{vid} was skipped because it has {len(pred_index)} pred_captions")
                continue
            
            if feature == "resnet":
                vis_feat = np.load(
                    f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_resnet.npy")
            elif feature == "bni":
                vis_feat = np.load(
                    f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_bn.npy")
            else:
                raise ValueError

            logger.debug(vid)
            logger.debug(pred_index)
            logger.debug(list(zip(*np.where(frame_data_all == 1))))
            logger.debug(vis_feat.shape)

            assert vis_feat.shape[0] == frame_data_all.shape[1], f"{vid}: {vis_feat.shape[0]}, {frame_data_all.shape[1]}"
            # p_comb, g_comb, sim_value = eval_func(
            #     pred_index, vis_feat, frame_data_all)
            # p_comb = p_comb.tolist()
            # g_comb = g_comb.tolist()

            # logger.debug(f"sim_value: {sim_value}")

            # assert len(g_comb) == len(p_comb), f"{len(g_comb)}, {len(p_comb)}"
            # assert len(p_comb) == len(pred_captions[vid]), f"{len(p_comb)}, {len(pred_captions[vid])}"
            # assert len(pred_index) == len(g_comb), f"{len(pred_index)}, {len(g_comb)}"

            # #captionとp_combのアライメントをとる
            logger.debug("======")
            logger.debug(pred_index)
            logger.debug(pred_captions[vid])
            # _p_comb = copy.deepcopy(p_comb)
            # sort_order = dict()
            # for i in range(len(pred_index)):
            #     index = _p_comb.index(pred_index[i])
            #     sort_order[pred_captions[vid][i]] = index
            #     _p_comb[index] = -1
            # pred_captions[vid] = sorted(pred_captions[vid], key=lambda x: sort_order[x])


            p_comb, g_comb, sim_value, selected_pred_captions, selected_gold_captions = eval_frame_match_multiple(pred_index, vis_feat, frame_data_all, pred_captions[vid], gold_captions[vid])
            selected_pred_index = copy.deepcopy(p_comb)

            if not (g_comb is not None and p_comb is not None and selected_pred_captions is not None):
                logger.info(f"{vid} is skipped because of invalid data")
                continue


            assert len(g_comb) == len(p_comb), f"{len(g_comb)}, {len(p_comb)}"
            assert len(p_comb) == len(selected_pred_captions), f"{len(p_comb)}, {len(selected_pred_captions)}"
            assert len(g_comb) == len(selected_gold_captions), f"{len(g_comb)}, {len(selected_gold_captions)}"


            tmp = list(zip(g_comb, p_comb, selected_pred_captions, selected_gold_captions))
            tmp = sorted(tmp, key=lambda x:x[0])
            g_comb, p_comb, p_comb_captions, g_comb_captions = list(zip(*tmp))
            logger.debug(p_comb)
            logger.debug(g_comb_captions)

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
        pickle.dump(df, open(output_file,'wb'))

    def eval_frame_match_using_val1val2_by_selected_numbers(self, dev_dataloader, oracle_test_dataloader, selected_test_dataloader, max_captions, output_file_name, feature, is_random=False, use_sub_movie=False, alpha=1.0):
        # self.tmp(dev_dataloader, oracle_test_dataloader, max_captions, output_file_name, feature, is_random=False, use_sub_movie=False)
        # assert False, "you should commentout here"
        test_golds, test_gold_captions,_ = self.create_golds_captions(oracle_test_dataloader)
        preds, _, test_pred_captions, _ = self.create_preds_golds_captions(selected_test_dataloader, is_random, use_sub_movie)
        _, dev_golds, _, dev_gold_captions = self.create_preds_golds_captions(dev_dataloader, is_random, use_sub_movie)

        # # 4こに限定する
        # preds, test_golds, test_pred_captions, test_gold_captions = self.create_preds_golds_captions(oracle_test_dataloader, is_random, use_sub_movie)
        # _, dev_golds, _, dev_gold_captions = self.create_preds_golds_captions(dev_dataloader, is_random, use_sub_movie)

        # for vid in preds.keys():
        #     preds[vid] = preds[vid][:4]
        #     test_pred_captions[vid] = test_pred_captions[vid][:4]

        # devとtestのgoldsをまとめる
        golds = defaultdict(list)
        gold_captions = defaultdict(list)
        test_vids_set = set(test_golds.keys())
        dev_vids_set = set(dev_golds.keys())

        for vid in dev_vids_set & test_vids_set:
            if vid in dev_vids_set:
                golds[vid].extend(dev_golds[vid])
                gold_captions[vid].extend(dev_gold_captions[vid])
            if vid in test_vids_set:
                golds[vid].extend(test_golds[vid])
                gold_captions[vid].extend(test_gold_captions[vid])

        
        output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}_using_val1val2_selected_numbers.pickle"
        output_df = defaultdict(list)
        for vid in preds.keys():
            pred_index = np.array(preds[vid])
            frame_data_all = np.array(golds[vid])

            if max_captions < len(pred_index):
                logger.info(f"{vid} caption num change {len(pred_index)} -> {max_captions}")
                # continue
                # max_captionsに合わせる
                pred_index = pred_index[:max_captions]
                test_pred_captions[vid] = test_pred_captions[vid][:max_captions]
                # assert False, f"{vid} was skipped because it has {len(pred_index)} captions"
            
            if feature == "resnet":
                vis_feat = np.load(
                    f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_resnet.npy")
            elif feature == "bni":
                vis_feat = np.load(
                    f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_bn.npy")
            else:
                raise ValueError
            logger.debug("======= start =========")
            logger.debug(vid)
            logger.debug(f"pred_index {pred_index}")
            logger.debug(f"captions {test_pred_captions[vid]}")
            logger.debug(f"gold index {list(zip(*np.where(frame_data_all == 1)))}")
            logger.debug(vis_feat.shape)

            assert vis_feat.shape[0] == frame_data_all.shape[1], f"{vid}: {vis_feat.shape[0]}, {frame_data_all.shape[1]}"

            from key_frame_captioning.eval.eval_frame_match_multiple import eval_frame_match_multiple
            p_comb, g_comb, sim_value, selected_pred_captions, selected_gold_captions = eval_frame_match_multiple(pred_index, vis_feat, frame_data_all, test_pred_captions[vid], gold_captions[vid], alpha)
            selected_pred_index = copy.deepcopy(p_comb)
                
            if not (g_comb is not None and p_comb is not None and selected_pred_captions is not None and selected_pred_index is not None):
                logger.info(f"{vid} is skipped because we can't select invalid frame g_comb:{g_comb}, p_comb:{p_comb} selectes_p_cap:{selected_pred_captions} selected_p_idx:{selected_pred_index}")
                continue

            gold_frame_idx_list = list(zip(*np.where(frame_data_all == 1)))

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

    def eval_frame_match_using_val1val2(self, dev_dataloader, test_dataloader, max_captions, select_captions, output_file_name, feature, is_random=False, use_sub_movie=False, alpha=1):
            preds, test_golds, test_pred_captions, test_gold_captions = self.create_preds_golds_captions(test_dataloader, is_random, use_sub_movie)
            _, dev_golds, _, dev_gold_captions = self.create_preds_golds_captions(dev_dataloader, is_random, use_sub_movie)

            # devとtestのgoldsをまとめる
            golds = defaultdict(list)
            gold_captions = defaultdict(list)
            test_vids_set = set(test_golds.keys())
            dev_vids_set = set(dev_golds.keys())

            for vid in dev_vids_set & test_vids_set:
                if vid in dev_vids_set:
                    golds[vid].extend(dev_golds[vid])
                    gold_captions[vid].extend(dev_gold_captions[vid])
                if vid in test_vids_set:
                    golds[vid].extend(test_golds[vid])
                    gold_captions[vid].extend(test_gold_captions[vid])

            
            output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}_using_val1val2.pickle"
            output_df = defaultdict(list)
            for vid in preds.keys():
                pred_index = np.array(preds[vid])
                frame_data_all = np.array(golds[vid])

                
                if max_captions < len(pred_index):
                    logger.info(f"{vid} caption num change {len(pred_index)} -> {max_captions}")
                    # continue
                    # max_captionsに合わせる
                    pred_index = pred_index[:max_captions]
                    test_pred_captions[vid] = test_pred_captions[vid][:max_captions]
                
                if feature == "resnet":
                    vis_feat = np.load(
                        f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_resnet.npy")
                elif feature == "bni":
                    vis_feat = np.load(
                        f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_bn.npy")
                else:
                    raise ValueError
                logger.debug("======= start =========")
                logger.debug(vid)
                logger.debug(f"pred_index {pred_index}")
                logger.debug(f"captions {test_pred_captions[vid]}")
                logger.debug(f"gold index {list(zip(*np.where(frame_data_all == 1)))}")
                logger.debug(vis_feat.shape)

                assert vis_feat.shape[0] == frame_data_all.shape[1], f"{vid}: {vis_feat.shape[0]}, {frame_data_all.shape[1]}"

                from key_frame_captioning.eval.eval_frame_match_multiple import eval_frame_match_multiple
                p_comb, g_comb, sim_value, selected_pred_captions, selected_gold_captions = eval_frame_match_multiple(pred_index, vis_feat, frame_data_all, test_pred_captions[vid], gold_captions[vid], alpha)
                selected_pred_index = copy.deepcopy(p_comb)
                    

                if not (g_comb is not None and p_comb is not None and selected_pred_captions is not None and selected_pred_index is not None):
                    logger.info(f"{vid} is skipped because we can't select invalid frame g_comb:{g_comb}, p_comb:{p_comb} selectes_p_cap:{selected_pred_captions} selected_p_idx:{selected_pred_index}")
                    continue

                ##frame_data_allからindexを生成したものとgold_captionの組み合わせ　selected_gold_captionsとg_combの組み合わせが有効なものなのかどうかを確かめる処理を記載
                # 何が知りたいのか
                # 元々のgold indexとcaptionの組み合わせに対してselectされたcaptionの組み合わせが一致しているのかの確認
                gold_frame_idx_list = list(zip(*np.where(frame_data_all == 1)))

                logger.debug(f"sim_value: {sim_value}")

                assert len(g_comb) == len(p_comb), f"{len(g_comb)}, {len(p_comb)}"
                assert len(p_comb) == len(selected_pred_captions), f"{len(p_comb)}, {len(selected_pred_captions)}"
                assert len(selected_pred_index) == len(g_comb), f"{len(selected_pred_index)}, {len(g_comb)}"

                #captionとp_combのアライメントをとる
                # logger.debug("=== selected ===")
                # logger.debug(f"selected_pred_index {selected_pred_index}")
                # logger.debug(f"selected_pred_captions {selected_pred_captions}")
                # _p_comb = copy.deepcopy(p_comb)
                # sort_order = dict()
                # for i in range(len(selected_pred_index)):
                #     index = _p_comb.index(selected_pred_index[i])
                #     sort_order[selected_pred_captions[i]] = index
                #     _p_comb[index] = -1
                # selected_pred_captions = sorted(selected_pred_captions, key=lambda x: sort_order[x])

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

    def eval_acc_loss(self,dataloader, output_file_name=None, is_random=False):
        loss_list = []
        acc_cnt = 0
        total = 0
        for b_resnet, b_bni, b_sent_ids, b_gold_labels, b_frame_masks, b_sent_mask, _, b_common_label, *_ in dataloader:
            b_resnet = b_resnet.to(self.device)
            b_bni = b_bni.to(self.device)
            b_sent_ids = b_sent_ids.to(self.device)
            b_gold_labels = b_gold_labels.to(self.device)
            b_sent_mask = b_sent_mask.to(self.device)
            b_frame_masks = b_frame_masks.to(self.device)
            b_common_label = b_common_label.to(self.device)

            if is_random:
                index = torch.LongTensor([torch.randint(b_frame_masks[i].sum(), (1,)).squeeze(0).item() for i in range(len(b_resnet))]).view(-1,1).to(self.device)
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(b_resnet, b_bni, b_sent_ids, b_frame_masks, b_sent_mask)
                    _, index = outputs.max(-1)
                    loss = self.criterion(outputs, b_gold_labels, b_common_label, b_frame_masks)
                    loss_list.append(loss.item())
                    _, index = torch.max(outputs, dim=-1)
                    index = index.view(-1,1)
            acc = torch.gather(b_gold_labels, -1, index).sum().item()

            total += b_resnet.size(0)
            acc_cnt += acc

        accuracy = acc_cnt / total
        avg_loss = sum(loss_list) / total

        output = {"accuracy": accuracy, "loss": avg_loss}
        logger.info(output)
        if output_file_name is not None:
            output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}.json"
            json.dump(output, open(output_file,'w'))


    def eval_frame_match_version3(self, dev_dataloader, oracle_test_dataloader, selected_test_dataloader, max_captions, output_file_name, feature, method, is_random=False, use_sub_movie=False, alpha=1.0, is_predict_frame=True, selected_alpha=0.5):
        #assert False #why assert false
        test_golds, test_gold_captions, _ = self.create_golds_captions(oracle_test_dataloader)
        preds, _, test_pred_captions, _, test_pred_timestamps, test_pred_frame_score, test_pred_caption_score = self.create_preds_golds_captions(selected_test_dataloader, is_random, use_sub_movie, is_predict_frame)
        # _, dev_golds, _, dev_gold_captions, *_ = self.create_preds_golds_captions(dev_dataloader, is_random, use_sub_movie)
        dev_golds, dev_gold_captions, _ = self.create_golds_captions(dev_dataloader)

        # devとtestのgoldsをまとめる
        golds = defaultdict(list)
        gold_captions = defaultdict(list)
        test_vids_set = set(test_golds.keys())
        dev_vids_set = set(dev_golds.keys())
        pred_vids_set = set(preds.keys())
        # vid_set = (test_vids_set | dev_vids_set) & pred_vids_set
        vid_set = pred_vids_set
        
        for vid in dev_vids_set | test_vids_set:
            if vid in dev_vids_set:
                golds[vid].extend(dev_golds[vid])
                gold_captions[vid].extend(dev_gold_captions[vid])
            if vid in test_vids_set:
                golds[vid].extend(test_golds[vid])
                gold_captions[vid].extend(test_gold_captions[vid])

        
        output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"
        output_json_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}.json"
        
        output_json = defaultdict(list)
        output_df = defaultdict(list)

        # json fileを生成する
        for vid in vid_set:
            assert len(test_pred_timestamps[vid]) == len(test_pred_captions[vid]) and  len(test_pred_captions[vid]) == len(preds[vid])
            for i in range(len(test_pred_timestamps[vid])):
                out = dict()
                out["timestamp"] = copy.deepcopy(test_pred_timestamps[vid][i])
                out["frame_idx"] = preds[vid][i]
                out["frame_sec"] = preds[vid][i] * 0.5 # hard coding!!!!!!!!!
                out["frame_score"] = test_pred_frame_score[vid][i]
                out["caption_score"] = test_pred_caption_score[vid][i]
                out["caption"] = test_pred_captions[vid][i]

                assert out["frame_score"] is not None, f'''{vid}:frame_score is None'''
                assert out["caption_score"] is not None, f'''{vid}:caption_score is None'''
                out["score"] = out["frame_score"] * selected_alpha + out["caption_score"] * (1 - selected_alpha)

                output_json[vid].append(copy.deepcopy(out))
            output_json[vid].sort(reverse=True,key=lambda x:x["score"])
        json.dump(output_json, open(output_json_file,'w'), indent=2)                

        for vid in vid_set:
            pred_index = np.array(preds[vid])
            frame_data_all = np.array(golds[vid])

            if max_captions < len(pred_index):
                logger.info(f"{vid} caption num change {len(pred_index)} -> {max_captions}")
                # continue
                # max_captionsに合わせる
                #candidateの形式に直す
                
                if method == "greedy":
                    selected_data = greedy(output_json[vid], max_captions)
                elif method == "dp":
                    selected_data = dp(output_json[vid], max_captions)
                elif method == "random":
                    selected_data = random_sampling(output_json[vid], max_captions)
                else:
                    raise ValueError

                if selected_data is None:
                    continue
                
                
                pred_index = [d["frame_idx"] for d in selected_data]
                test_pred_captions[vid] = [d["caption"] for d in selected_data]
                # pred_index = pred_index[:max_captions]
                # test_pred_captions[vid] = test_pred_captions[vid][:max_captions]
                # assert False, f"{vid} was skipped because it has {len(pred_index)} captions"
            
            if feature == "resnet":
                vis_feat = np.load(
                    f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_resnet.npy")
            elif feature == "bni":
                vis_feat = np.load(
                    f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_bn.npy")
            else:
                raise ValueError

            # skip_flag = False
            # for pi in pred_index:
            #     if vis_feat.shape[0] <= pi:
            #         skip_flag = True
            #         # logger.warning(f"invalid pred index: {vid} pred index:{pi}, video frame:{vis_feat.shape[0]}")
            # if skip_flag:
            #     continue
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
                raise RuntimeError(
                    f"{vid} cause error because we can't select invalid frame g_comb:{g_comb}, p_comb:{p_comb} selectes_p_cap:{selected_pred_captions} selected_p_idx:{selected_pred_index}"
                )
            
            gold_frame_idx_list = list(zip(*np.where(frame_data_all == 1)))

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


    # def predict_version3(self, dev_dataloader, oracle_test_dataloader, selected_test_dataloader, max_captions, output_file_name, feature, is_random=False, use_sub_movie=False):
    #     test_golds, test_gold_captions = self.create_golds_captions(oracle_test_dataloader)
    #     preds, _, test_pred_captions, _, test_pred_timestamps = self.create_preds_golds_captions(selected_test_dataloader, is_random, use_sub_movie, is_predict_frame)
    #     _, dev_golds, _, dev_gold_captions, dev_gold_timestamps = self.create_preds_golds_captions(dev_dataloader, is_random, use_sub_movie)

    #     # devとtestのgoldsをまとめる
    #     golds = defaultdict(list)
    #     gold_captions = defaultdict(list)
    #     test_vids_set = set(test_golds.keys())
    #     dev_vids_set = set(dev_golds.keys())
    #     pred_vids_set = set(preds.keys())
    #     vid_set = test_vids_set & dev_vids_set & pred_vids_set

    #     output_

    #     for vid in dev_vids_set & test_vids_set:
    #         if vid in dev_vids_set:
    #             golds[vid].extend(dev_golds[vid])
    #             gold_captions[vid].extend(dev_gold_captions[vid])
    #         if vid in test_vids_set:
    #             golds[vid].extend(test_golds[vid])
    #             gold_captions[vid].extend(test_gold_captions[vid])

        
    #     output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}_predict_version3.json"
    #     output_df = defaultdict(list)
    #     for vid in vid_set:
        
    def eval_frame_match_version3_predict_only(self, reference_dataloader, pred_dataloader, output_file_name, is_random=False, use_sub_movie=False, is_predict_frame=True):
        test_golds, test_gold_captions, test_gold_timestamps = self.create_golds_captions(
            reference_dataloader
        )
        preds, _, test_pred_captions, _, test_pred_timestamps, test_pred_frame_score, test_pred_caption_score = self.create_preds_golds_captions(
            pred_dataloader,
            is_random,
            use_sub_movie,
            is_predict_frame
        )
        
        # devとtestのgoldsをまとめる
        golds = defaultdict(list)
        gold_captions = defaultdict(list)
        test_vids_set = set(test_golds.keys())
        pred_vids_set = set(preds.keys())
        gold_timestamps = defaultdict(list)
        vid_set = pred_vids_set

        # dev_vids_set & test_vids_setをしているのになぜif文で条件分岐している？
        for vid in test_vids_set:
            golds[vid].extend(test_golds[vid])
            gold_captions[vid].extend(test_gold_captions[vid])
            gold_timestamps[vid].extend(test_gold_timestamps[vid])
        
        output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"
        output_json_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}.json"

        output_json = defaultdict(list)

        # json fileを生成する
        for vid in vid_set:
            assert len(test_pred_timestamps[vid]) == len(test_pred_captions[vid]) and len(test_pred_captions[vid]) == len(preds[vid])
            for i in range(len(test_pred_timestamps[vid])):
                out = dict()
                out["timestamp"] = copy.deepcopy(test_pred_timestamps[vid][i])
                out["frame_idx"] = preds[vid][i]
                out["frame_sec"] = self.pp.duration_sec2trainvalsec(vid, self.pp.index2sec(vid, preds[vid][i])) # trainvalの秒数に合わせて保存する
                out["frame_score"] = test_pred_frame_score[vid][i]
                out["caption_score"] = test_pred_caption_score[vid][i]
                out["caption"] = test_pred_captions[vid][i]

                # assert out["frame_score"] is not None, f'''{vid}:frame_score is None'''
                # assert out["caption_score"] is not None, f'''{vid}:caption_score is None'''
                # out["score"] = out["frame_score"] * selected_alpha + out["caption_score"] * (1-selected_alpha)

                output_json[vid].append(copy.deepcopy(out))
        json.dump(output_json, open(output_json_file,'w'), indent=2)                

        pickle_data = dict()
        pickle_data["golds"] = copy.deepcopy(golds)
        pickle_data["preds"] = copy.deepcopy(preds)
        pickle_data["gold_captions"] = copy.deepcopy(gold_captions)
        pickle_data["test_pred_captions"] = copy.deepcopy(test_pred_captions)
        pickle_data["gold_timestamps"] = copy.deepcopy(gold_timestamps)

        pickle.dump(pickle_data, open(output_file, 'wb'))


    def eval_frame_match_version3_eval_only(self, max_captions, output_file_name, feature, method, pickle_path, json_path, alpha=1.0, selected_alpha=0.5):

        _output_json = json.load(open(json_path))
        output_json = defaultdict(list)
        pickle_data = pickle.load(open(pickle_path, 'rb'))
        golds = pickle_data["golds"]
        preds = pickle_data["preds"]
        gold_captions = pickle_data["gold_captions"]
        test_pred_captions = pickle_data["test_pred_captions"]
        gold_timestamps = pickle_data["gold_timestamps"]
        vid_set = _output_json.keys()

        output_df = defaultdict(list)
        output_file = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"
        gold_json_path = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}_gold.json"
        pred_json_path = f"{self.root_dir}/{self.eval_config['save_dir']['eval_dir']}/{output_file_name}_pred.json"
        gold_json_data = defaultdict(list)
        pred_json_data = dict()

        # scoreの重みつけをする
        for vid in vid_set:
            out = []
            for i in range(len(_output_json[vid])):
                score = _output_json[vid][i]["frame_score"] * selected_alpha + _output_json[vid][i]["caption_score"] * (1 - selected_alpha)
                out.append(copy.deepcopy(_output_json[vid][i]))
                out[i]["score"] = score
            output_json[vid] = copy.deepcopy(out)
            output_json[vid].sort(reverse=True,key=lambda x:x["score"])

        # scoreをもとにキーフレームを選択する
        for vid in vid_set:
            pred_index = np.array(preds[vid])
            frame_data_all = np.array(golds[vid])

            if max_captions < len(pred_index):
                logger.info(f"{vid} caption num change {len(pred_index)} -> {max_captions}")
                # continue
                # max_captionsに合わせる
                #candidateの形式に直す
                
                if method == "greedy":
                    selected_data = greedy(output_json[vid], max_captions)
                elif method == "dp":
                    selected_data = dp(output_json[vid], max_captions)
                elif method == "random":
                    selected_data = random_sampling(output_json[vid], max_captions)
                else:
                    raise ValueError

                if selected_data is None:
                    continue
                
                pred_index = [d["frame_idx"] for d in selected_data]
                test_pred_captions[vid] = [d["caption"] for d in selected_data]
                # pred_index = pred_index[:max_captions]
                # test_pred_captions[vid] = test_pred_captions[vid][:max_captions]
                # assert False, f"{vid} was skipped because it has {len(pred_index)} captions"
            
            if feature == "resnet":
                vis_feat = np.load(
                    f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_resnet.npy")
            elif feature == "bni":
                vis_feat = np.load(
                    f"{self.eval_config['frame_match_feature_root']}/{self.image_set}/{vid}_bn.npy")
            else:
                raise ValueError

            # skip_flag = False
            # for pi in pred_index:
            #     if vis_feat.shape[0] <= pi:
            #         skip_flag = True
            #         # logger.warning(f"invalid pred index: {vid} pred index:{pi}, video frame:{vis_feat.shape[0]}")
            # if skip_flag:
            #     continue
            pred_index = [min(vis_feat.shape[0]-1, p) for p in pred_index]

            # jsonに格納
            # pred_json_data[vid] = [self.pp.duration_sec2trainvalsec(vid, self.pp.index2sec(vid, pi)) for pi in pred_index]
            pred_json_data[vid] = [pi for pi in pred_index]

            gold_frame_idx_list = list(zip(*np.where(frame_data_all == 1)))
            for i in range(len(gold_timestamps[vid])):
                # start_sec = self.pp.duration_sec2trainvalsec(vid, gold_timestamps[vid][i][0])
                # end_sec = self.pp.duration_sec2trainvalsec(vid, gold_timestamps[vid][i][1])
                # gold_json_data[vid].append({"start":start_sec, "end":end_sec, "ann_secs":[]})
                gold_json_data[vid].append({"start":self.pp.sec2index(vid, gold_timestamps[vid][i][0]), "end":self.pp.sec2index(vid, gold_timestamps[vid][i][1]), "ann_secs":[]})
            for index, frame_idx in gold_frame_idx_list:
                # frame_sec = self.pp.duration_sec2trainvalsec(vid,self.pp.index2sec(vid, frame_idx))
                # gold_json_data[vid][index]["ann_secs"].append(frame_sec)
                gold_json_data[vid][index]["ann_secs"].append(int(frame_idx))

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
                raise ValueError(f"{vid} is skipped because we can't select invalid frame g_comb:{g_comb}, p_comb:{p_comb} selectes_p_cap:{selected_pred_captions} selected_p_idx:{selected_pred_index}")

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
        
        json.dump(pred_json_data, open(pred_json_path,'w'),indent=2)
        json.dump(gold_json_data, open(gold_json_path,'w'),indent=2)
