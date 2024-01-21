
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from collections import defaultdict
import pandas as pd
import random
import os
import copy
from logzero import logger
import math
# dataloader for training
import sys
sys.path.append("..")
from preprocess.sec_preprocess import Preprocess

class ANetDataset(Dataset):
    def __init__(self,  train_config, tokenizer, pad_token_id, vocab, debug, image_set="training", is_sampling=False, use_sub_movie=False, dense_cap_file=None, v_sec_thresh=1.0, cos_sim_feature_name=None, use_only_movie_exist_vid=False):
        super(ANetDataset, self).__init__()

        self.pp = Preprocess()

        self.feature_root = train_config["feature_root"]

        self.image_set = image_set
        
        if image_set == 'training':
            self.tsv_mode = ['train']
        elif image_set == 'validation':
            self.tsv_mode = ['val_1']
        elif image_set == 'testing':
            self.image_set = "validation"
            self.tsv_mode = ['val_2']
        elif image_set == "validation_testing":
            self.tsv_mode = ["val_1", 'val_2']
            self.image_set = "validation"
        elif image_set == "fixed_eval":
            self.tsv_mode = None
            self.image_set = image_set
        else:
            raise ValueError

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.pad_token_id = pad_token_id

        self.debug = debug
        self.is_sampling = is_sampling

        self.use_sub_movie = use_sub_movie

        if dense_cap_file is not None:
            # densecapの結果を使用
            densecap_json_data = json.load(open(dense_cap_file))
            logger.info("densecap data was loaded")

        duration_df = pd.read_csv(train_config["dur_file"])
        duration_df.columns = ["vid", "v_sec", "frame_size"]
        
        self.ann = defaultdict(lambda: defaultdict(
            lambda: {"gold_caption": None, "pred_sentence": None, "ann_secs": [], "v_sec": -1, "label": None, "start_sec": None, "end_sec": None}))
        
        with open(train_config["dataset_file"], 'r') as f:
            for line in f:

                line = line.rstrip('\r\n')
                _, vid, _, _,  ann_sec, caption, _, start_sec, end_sec,  mode, v_sec = line.split('\t')
                start_sec, end_sec = self.pp.trainval2duration_sec(vid,float(start_sec)), self.pp.trainval2duration_sec(vid,float(end_sec))
                #captionが同じでsegmentが異なるというcaptionの対応するために修正
                key = f"{caption}_{start_sec}_{end_sec}"
                ann_sec = self.pp.trainval2duration_sec(vid,float(ann_sec))
                v_sec = float(v_sec)
                vid = vid[2:]  # "v_"を削除
                if (self.tsv_mode is not None) and (mode not in self.tsv_mode):
                    raise ValueError(f"{mode} is not in {self.tsv_mode}")
                
                #if debug and 100 <= len(self.ann) and vid not in self.ann:
                #    break

                # 一つのキャプションではstart sec とend secは一致している必要がある
                assert self.ann[vid][key]["start_sec"] is None or self.ann[vid][key]["start_sec"] == start_sec
                assert self.ann[vid][key]["end_sec"] is None or self.ann[vid][key]["end_sec"] == end_sec
                self.ann[vid][key]["ann_secs"].append(ann_sec)
                self.ann[vid][key]["start_sec"] = start_sec
                self.ann[vid][key]["end_sec"] = end_sec
                self.ann[vid][key]["gold_caption"] = caption
                self.ann[vid][key]["pred_sentence"] = self.tokenizer(caption)
                self.ann[vid][key]["v_sec"] = v_sec

        self.vids = []
        for vid in self.ann:
            if (duration_df["vid"] == vid).any():
                # 動画の総時間を取得
                v_sec = duration_df[duration_df["vid"] == vid]["v_sec"].iloc[0]
                # csv　fileとtsvファイルで動画の時間が異なっているものを除外
                _v_sec = [v["v_sec"] for v in self.ann[vid].values()][0]
                if v_sec_thresh <= abs(_v_sec  - v_sec):
                    raise ValueError(f"{vid} exceed time threshold({v_sec_thresh}): {v_sec}, {_v_sec}")
            else:
                raise ValueError(f"{vid} doesn't exist in duration file")
                continue

            # densecapの結果を使用
            if dense_cap_file is not None:
                _vid = f"v_{vid}"
                if _vid in densecap_json_data:
                    tmp = self.replace_by_densecap_data(vid, self.ann[vid], densecap_json_data[_vid])
                    self.ann[vid] = copy.deepcopy(tmp)
                else:
                    logger.warning(f"{vid} doesn't exist in densecap json file")
                    continue

            feature_path = f"{self.feature_root}/{self.image_set}/{vid}_resnet.npy"
            if os.path.isfile(feature_path):
                features = torch.from_numpy(np.load(feature_path)).float()
                frame_cnt = features.size(0)
            else:
                logger.warning(f"{feature_path} doesn't exist")
                continue
            
            movie_path = f"{train_config['movie_root']}/v_{vid}.mp4"
            if use_only_movie_exist_vid is True and not os.path.isfile(movie_path):
                logger.warning(f"{vid} doesn't exist in movie directory")
                continue

            common_label = None

            captions = tuple(self.ann[vid].keys())
            is_invalid_gold_label = False
            for caption in captions:
                values = self.ann[vid][caption]
                ann_secs = values["ann_secs"]

                if v_sec < values["start_sec"]:
                    logger.warning(f"{vid} start sec is over than total sec. {v_sec}, {values['start_sec']}")
                    del self.ann[vid][caption]
                    continue
                if v_sec < values["end_sec"]:
                    values["end_sec"] =  v_sec
                    
                assert float(f'{values["end_sec"]:.2f}') <= float(f'{v_sec:.2f}'), f'''{vid}:{values["end_sec"]}, {v_sec}'''

                # sort
                if self.ann[vid][caption]["ann_secs"] is not None:
                    self.ann[vid][caption]["ann_secs"].sort() 
                if cos_sim_feature_name is None:
                    # label, start_idx, end_idx = self.calc_label(v_sec, ann_secs, frame_cnt, train_config["label_offset"], values["start_sec"], values["end_sec"])
                    start_idx = self.pp.sec2index(vid, values["start_sec"])
                    end_idx = self.pp.sec2index(vid, values["end_sec"])
                    offset = train_config["label_offset"]
                    label = torch.FloatTensor([0 for _ in range(int(frame_cnt))])
                    if not dense_cap_file:
                        for frame_idx in map(lambda x:self.pp.sec2index(vid, x), ann_secs):
                            if frame_idx is not None:
                                label[max(0, frame_idx - offset):frame_idx+offset+1] = 1

                elif cos_sim_feature_name in ["resnet", "bni"]:
                    label, start_idx, end_idx = self.calc_cos_sim_label(v_sec, ann_secs, features, values["start_sec"], values["end_sec"])
                else:
                    raise ValueError(f"{cos_sim_feature_name} is invalid name")

                assert start_idx is not None and end_idx is not None, f"vid:{vid}, v sec:{v_sec}, frame cnt:{frame_cnt}, start sec:{values['start_sec']}, idx:{start_idx}, end sec:{values['end_sec']}, idx:{end_idx}"

                if common_label is None:
                    common_label = label.clone()
                else:
                    common_label += label.clone()
                    
                self.ann[vid][caption]["label"] = label
                self.ann[vid][caption]["start_idx"] = start_idx
                self.ann[vid][caption]["end_idx"] = end_idx

            if is_invalid_gold_label is False:
                self.vids.append(vid)

            for caption, values in self.ann[vid].items():
                self.ann[vid][caption]["common_label"] = torch.where(
                    common_label == 0, common_label, torch.ones_like(common_label))
        if not self.is_sampling:
            self.serialized_ann = []
            for vid in self.vids:
                for key, ann in self.ann[vid].items():

                    ann["vid"] = vid
                    ann["caption"] = self.get_caption_from_key(key)
                    # ann["caption"] = ann["gold_caption"]
                    self.serialized_ann.append(ann)

        logger.info(f"{len(self.vids)} data was loaded")

    def get_caption_from_key(self, key):
        # assert False, "do not use this funcrtion"
        return '_'.join(key.split("_")[:-2])


    # def replace_by_densecap_data(self, ann_data, densecap_data):
    #     """
    #     goldのデータをdensecapの出力データに置換する
    #     """
        
    #     ret = []
    #     for cap, v in islice(copy.deepcopy(ann_data).items(), len(densecap_data)):
    #         v["caption"] = cap
    #         ret.append(v)
    #     ret = list(sorted(ret, key=lambda x:x["start_sec"]))
        
    #     densecap_data = list(sorted(densecap_data, key=lambda x:x["timestamp"][0]))
    #     _ann_data = defaultdict(dict)
    #     for i in range(len(ret)):
    #         ret[i]["pred_sentence"] = self.tokenizer(densecap_data[i]['sentence'])
    #         ret[i]["caption"] = f"{densecap_data[i]['sentence']}_{densecap_data[i]['timestamp'][0]}_{densecap_data[i]['timestamp'][1]}"
    #         ret[i]["start_sec"] = densecap_data[i]["timestamp"][0]
    #         ret[i]["end_sec"] = densecap_data[i]["timestamp"][1]

    #         # version3 model用にデータを追加
    #         ret[i]["caption_score"] = densecap_data[i]["caption_score"]
    #         ret[i]["frame_score"] = densecap_data[i]["frame_score"] if "frame_score" in densecap_data[i] else None
    #         ret[i]["frame_idx"] = densecap_data[i]["frame_idx"] if "frame_idx" in densecap_data[i] else None

    #         for k, v in ret[i].items():
    #             if k == "caption":
    #                 continue
    #             _ann_data[ret[i]["caption"]][k] = v
    #     assert len(_ann_data) == len(densecap_data), f"{len(_ann_data)}, {len(densecap_data)}"
    #     return copy.deepcopy(_ann_data)
    def replace_by_densecap_data(self, vid, ann_data, densecap_data):
        """
        goldのデータをdensecapの出力データに置換する．両者のデータ数が同じである必要はない
        """
                
        densecap_data = list(sorted(densecap_data, key=lambda x:x["timestamp"][0]))
        _ann_data = defaultdict(dict)
        for i in range(len(densecap_data)):
            ret = dict()
            ret["pred_sentence"] = self.tokenizer(densecap_data[i]['sentence'])
            ret["start_sec"] = self.pp.trainval2duration_sec(vid, densecap_data[i]["timestamp"][0])
            ret["end_sec"] = self.pp.trainval2duration_sec(vid, densecap_data[i]["timestamp"][1])
            ret["caption"] = f'''{densecap_data[i]['sentence']}_{ret["start_sec"]}_{ret["end_sec"]}'''
            ret["gold_caption"] = None
            ret["ann_secs"] = []

            # version3 model用にデータを追加
            ret["caption_score"] = densecap_data[i]["caption_score"]
            ret["frame_score"] = densecap_data[i]["frame_score"] if "frame_score" in densecap_data[i] else densecap_data[i]["segment_score"]
            ret["frame_idx"] = densecap_data[i]["frame_idx"] if "frame_idx" in densecap_data[i] else None

            for k, v in copy.deepcopy(ret).items():
                if k == "caption":
                    continue
                _ann_data[ret["caption"]][k] = v
            
        return copy.deepcopy(_ann_data)    


    # @classmethod
    # def identify_label_from_sec(cls, v_sec, frame_cnt, ann_secs):
    #     """ann_secsの秒数を表すlabel indexを特定"""
    #     ret = []
    #     frame_per_sec = v_sec / frame_cnt
    #     frame_idx = 0
    #     for ann_sec in sorted(ann_secs):
    #         while frame_idx < frame_cnt:
    #             now_sec = frame_per_sec*frame_idx
    #             upper = cls.ceil(now_sec + (frame_per_sec/2), 4)
    #             lower = cls.floor(now_sec - (frame_per_sec/2), 4)
    #             if lower <= ann_sec and ann_sec < upper:
    #                 ret.append(frame_idx)
    #                 break
    #             frame_idx += 1
    #     # annsecがframeないに治っていない時は最後のフレームに割り当てる
    #     while len(ret) != len(ann_secs):
    #         ret.append(frame_cnt - 1)
    #     assert len(ret) == len(ann_secs), f"{v_sec}, {frame_cnt}, {ann_secs}, {ret}"
    #     return ret


    # @classmethod
    # def create_start_end_idx_from_sec(cls, v_sec, frame_cnt, start_sec, end_sec):
    #     """start_idxとend_idxを計算"""
    #     frame_per_sec = v_sec / frame_cnt
    #     start_idx, end_idx = None, None
    #     if not (start_sec is None and end_sec is None):
    #         frame_idx = 0
    #         while frame_idx < frame_cnt:
    #             now_sec = frame_per_sec*frame_idx
    #             upper = cls.ceil(now_sec + (frame_per_sec/2), 4)
    #             lower = cls.floor(now_sec - (frame_per_sec/2), 4)
    #             # print(lower, upper, start_sec, end_sec, start_idx, end_idx)
    #             if lower <= start_sec and start_sec < upper:
    #                 start_idx = frame_idx

    #             if lower <= end_sec and end_sec < upper:
    #                 end_idx = frame_idx
    #             if start_idx is not None and end_idx is not None:
    #                 break
    #             frame_idx += 1
    #         # end_idxがv_secの中に収まっていない時がある
    #         if end_idx is None:
    #             end_idx = frame_cnt - 1
    #         if start_idx is None:
    #             start_idx = end_idx
    #     return start_idx, end_idx

    @staticmethod
    def regularize_tensor(x):
        min_x, _ = x.min(-1)
        max_x, _ = x.max(-1)
        regulxrized_x = (x - min_x)/(max_x - min_x)
        return regulxrized_x

    def calc_cos_sim_label(self, v_sec, ann_secs, features, start_sec=None, end_sec=None):
        cos_sim_list = torch.FloatTensor([])
        # for index in self.identify_label_from_sec(v_sec, features.size(0), ann_secs):
        for index in map(lambda x:self.pp.sec2index(x), ann_secs):
            cos_sim_list = torch.cat((cos_sim_list, torch.nn.functional.cosine_similarity(features, features[index].view(1, -1)).view(1, -1)), dim=0)
        ret, _ = cos_sim_list.max(0)
        ret[ret < 0] = 0.0 #類似度が0以下の部分は0に置換
        ret = self.regularize_tensor(ret)
        assert features.size(0) == ret.size(0)
        # start_idx, end_idx = cls.create_start_end_idx_from_sec(v_sec, features.size(0), start_sec, end_sec)
        start_idx = self.pp.sec2index(start_sec)
        end_idx = self.pp.sec2index(end_sec)
        return ret, start_idx, end_idx


    # @classmethod
    # def calc_label(cls, v_sec, ann_secs, frame_cnt, offset, start_sec=None, end_sec=None):
    #     label = torch.FloatTensor([0 for _ in range(int(frame_cnt))])
    #     frame_per_sec = v_sec / frame_cnt
    #     frame_idx = 0

    #     if ann_secs is not None:
    #         for ann_sec in sorted(ann_secs):
    #             while frame_idx < frame_cnt:
    #                 now_sec = frame_per_sec*frame_idx
    #                 upper = cls.ceil(now_sec + (frame_per_sec/2), 4)
    #                 lower = cls.floor(now_sec - (frame_per_sec/2), 4)
    #                 if lower <= ann_sec and ann_sec < upper:
    #                     label[max(0, frame_idx - offset):frame_idx+offset+1] = 1
    #                     break
    #                 frame_idx += 1
        
    #     start_idx, end_idx = cls.create_start_end_idx_from_sec(v_sec, frame_cnt, start_sec, end_sec)

    #     return label, start_idx, end_idx




    def sampling(self, vid):
        values = list(self.ann[vid].values())
        captions = list(self.ann[vid].keys())
        index = random.randint(0, len(values)-1)
        caption = self.get_caption_from_key(captions[index])
        # caption = [a["caption"] for a in self.ann[vid]]
        ann = values[index]
        ann["caption"] = caption
        return ann
        
    @staticmethod
    def floor(value, digit):
        return math.floor(value*(10**digit)) / (10**digit)

    @staticmethod
    def ceil(value, digit):
        return math.ceil(value*(10**digit)) / (10**digit)

    @classmethod
    def triming_tensor(self, tensor, start_idx, end_idx):
        assert len(tensor) == len(start_idx) and len(start_idx) == len(end_idx), f"{len(tensor)}, {len(start_idx)}, {len(end_idx)}"
        ret_tensor = []
        for index in range(len(start_idx)):
            _tensor = tensor[index].clone()
            ret_tensor.append(_tensor[start_idx[index]:end_idx[index]+1])
            assert ret_tensor[-1].size(0) == end_idx[index] - start_idx[index] + 1, f'''{_tensor.size(0)} {ret_tensor[-1].size(0)}, {end_idx[index]} {start_idx[index]}'''
        return ret_tensor


    def __len__(self):
        if self.is_sampling:
            return len(self.vids)
        else:
            return len(self.serialized_ann)

    def __getitem__(self, idx):
        """
        データを返す

        Parameters
        ----------
        idx : int
            self.vidsのリストのindex

        Returns
        -------
        resnet_feat : FloatTensor
            resnetの特徴量 (frame_cnt, resnetの特徴量次元)
        bni_feat : FloatTensor
            bniの特徴量 (frame_cnt, bniの特徴量次元)   
        sentence : LongTensor
            キャプションをid化したもの (単語数の次元)
        frame_cnt : int
            フレーム数
        gold_label : FloatTensor
            キャプションがつくフレームに1が立っているtensor (frame_cnt数の次元)　複数の場所に1がついている可能性あり？
        """

        if self.is_sampling:
            vid = self.vids[idx]
            ann = self.sampling(vid)
            gold_caption, pred_sentence, gold_label, common_label, start_idx, end_idx = ann["gold_caption"], ann["pred_sentence"], ann["label"], ann["common_label"], ann["start_idx"], ann["end_idx"]
            ann_secs = ann["ann_secs"]
            caption = ann["caption"]

            frame_idx = ann["frame_idx"] if "frame_idx" in ann else None
            frame_score = ann["frame_score"] if "frame_score" in ann else None
            caption_score = ann["caption_score"] if "caption_score" in ann else None

            start_sec = ann["start_sec"]
            end_sec = ann["end_sec"]
        else:
            vid = self.serialized_ann[idx]["vid"]
            gold_caption = self.serialized_ann[idx]["gold_caption"]
            pred_sentence = self.serialized_ann[idx]["pred_sentence"]
            gold_label = self.serialized_ann[idx]["label"]
            common_label = self.serialized_ann[idx]["common_label"]
            start_idx = self.serialized_ann[idx]["start_idx"]
            end_idx = self.serialized_ann[idx]["end_idx"]
            start_sec = self.serialized_ann[idx]["start_sec"]
            end_sec = self.serialized_ann[idx]["end_sec"]
            ann_secs = self.serialized_ann[idx]["ann_secs"]
            caption = self.serialized_ann[idx]["caption"]
            # print(self.serialized_ann[idx].keys())
            frame_idx = self.serialized_ann[idx]["frame_idx"] if "frame_idx" in self.serialized_ann[idx] else None
            frame_score = self.serialized_ann[idx]["frame_score"] if "frame_score" in  self.serialized_ann[idx] else None
            caption_score = self.serialized_ann[idx]["caption_score"] if "caption_score" in self.serialized_ann[idx] else None


        resnet_feat = torch.from_numpy(
            np.load(f"{self.feature_root}/{self.image_set}/{vid}_resnet.npy")).float()
        bni_feat = torch.from_numpy(
            np.load(f"{self.feature_root}/{self.image_set}/{vid}_bn.npy")).float()
        frame_cnt = bni_feat.size(0)

        sentence_len = len(pred_sentence)

        assert gold_label.size() == common_label.size(), f"{gold_label.size()}, {common_label.size()}"

        return resnet_feat, bni_feat, pred_sentence, frame_cnt, sentence_len,  gold_label, vid, common_label, start_idx, end_idx, ann_secs, caption, gold_caption, frame_idx, frame_score, caption_score, start_sec, end_sec

    def collate_fn(self, batch):
        """
        paddingする

        Parameters
        ----------
        batch : List
            バッチ

        Returns
        -------
        resnet_feat : FloatTensor
            resnetの特徴量 (batch_size, バッチ内最大frame数, resnetの特徴量次元)
        bni_feat : FloatTensor
            bniの特徴量 (batch_size, バッチ内最大frame数, bniの特徴量次元)   
        sentence : LongTensor
            キャプションをid化したもの (batch_size, バッチ内最大単語数の次元)
        frame_mask : LongTensor
            padding以外に1がたったmask(文長) (batch_size, バッチ内最大単語数の次元)
        gold_label : FloatTensor
            キャプションがつくフレームに1が立っているtensor (batch_size, バッチ内最大frame数)　複数の場所に1がついている可能性あり？
        frame_mask : LongTensor
            padding以外に1がたったmask(フレーム長) (batch_size, バッチ内最大frame数)
        """

        resnet_feat, bni_feat, pred_sentence, frame_cnt, sentence_len,  gold_label, vid, common_label, start_idx, end_idx, ann_secs, caption, gold_caption, frame_idx, frame_score, caption_score, start_sec, end_sec = zip(*batch)

        # batchをトリミングする
        original_gold_label = copy.deepcopy(gold_label)
        original_frame_cnt = copy.deepcopy(frame_cnt)
        if self.use_sub_movie:
            frame_cnt = []
            for sidx, eidx in zip(start_idx, end_idx):
                frame_cnt.append(eidx - sidx + 1)

            resnet_feat = self.triming_tensor(resnet_feat, start_idx, end_idx)
            bni_feat = self.triming_tensor(bni_feat, start_idx, end_idx)
            gold_label = self.triming_tensor(gold_label, start_idx, end_idx)
            common_label = self.triming_tensor(common_label, start_idx, end_idx)

            for index in range(len(frame_cnt)):
                assert frame_cnt[index] == resnet_feat[index].size(0), f"{vid[index]}:{frame_cnt[index]}, {resnet_feat[index].size(0)}, {start_idx[index]}, {end_idx[index]}"
                assert gold_label[index].size(0) == bni_feat[index].size(0), f"{vid[index]}:{gold_label[index].size(0)}, {bni_feat[index].size(0)}"

        resnet_feat = pad_sequence(resnet_feat, batch_first=True)
        bni_feat = pad_sequence(bni_feat, batch_first=True)
        pred_sentence = pad_sequence(pred_sentence, batch_first=True,
                                padding_value=self.pad_token_id)
        gold_label = pad_sequence(gold_label, batch_first=True)
        original_gold_label = pad_sequence(original_gold_label, batch_first=True)
        common_label = pad_sequence(common_label, batch_first=True)

        sentence_mask = torch.ones(pred_sentence.size()).long()
        for i in range(len(sentence_mask)):
            sentence_mask[i, sentence_len[i]:] = 0

        frame_mask = torch.ones(bni_feat.size(0), bni_feat.size(1)).long()
        for i in range(len(frame_mask)):
            frame_mask[i, frame_cnt[i]:] = 0

        assert gold_label.size(1) == resnet_feat.size(
            1),  f"{gold_label.size(1)}, {resnet_feat.size(1)}"

        return resnet_feat, bni_feat, pred_sentence, gold_label, frame_mask, sentence_mask, vid, common_label, (original_gold_label, start_idx, end_idx, original_frame_cnt), (ann_secs, caption), gold_caption, (frame_idx, frame_score, caption_score), (start_sec, end_sec)
