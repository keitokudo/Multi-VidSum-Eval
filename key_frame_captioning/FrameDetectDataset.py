
import torch
import torchtext
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
from pprint import pprint
from Dataset import ANetDataset

# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from preprocess.video_preprocess import VideoConverter


def d():
    return defaultdict(dd)

def dd():
    return {"sentence": None, "ann_secs": [], "v_sec": -1, "labels":[]}


class FrameANetDataset(ANetDataset):
    # vc = VideoConverter(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    def __init__(self, train_config=None, tokenizer=None, pad_token_id=None, vocab=None, debug=None, image_set="training", is_sampling=False, dense_cap_file=None, v_sec_thresh=1.0, sampling_sec=0.5):
        # super(ANetDataset, self).__init__(train_config, tokenizer, pad_token_id, vocab, debug, image_set, is_sampling, False, dense_cap_file, v_sec_thresh)
        super(ANetDataset, self).__init__()
        self.use_sub_movie = False

        self.train_config = train_config
        self.feature_root = train_config["feature_root"]

        # annを修正
        self.image_set = image_set

        # testの時はval1とval2どちらも使用する
        if image_set == 'training':
            self.tsv_mode = ['train']
        elif image_set == 'validation':
            self.tsv_mode = ['val_1']
        elif image_set == 'testing':
            self.image_set = "validation"
            self.tsv_mode = ['val_2']
        else:
            raise ValueError

        self.vocab = vocab
        self.pad_token_id = pad_token_id

        self.debug = debug
        self.is_sampling = is_sampling
        self.sampling_sec = sampling_sec

        if train_config is None:
            return

        duration_df = pd.read_csv(train_config["dur_file"])
        duration_df.columns = ["vid", "v_sec", "frame_size"]

        self.ann = defaultdict(d)

        with open(train_config["dataset_file"], 'r') as f:
            for line in f:

                line = line.rstrip('\r\n')
                _, vid, _, _,  ann_sec, caption, _, start_sec, end_sec,  mode, v_sec = line.split(
                    '\t')
                start_sec, end_sec = float(start_sec), float(end_sec)
                ann_sec = float(ann_sec)
                #captionが同じでsegmentが異なるというcaptionの対応するために修正
                key = f"{caption}_{start_sec}_{end_sec}"
                v_sec = float(v_sec)
                vid = vid[2:]  # "v_"を削除
                if mode not in self.tsv_mode:
                    continue
                
                if debug and 10 <= len(self.ann) and vid not in self.ann:
                    break

                # numpyファイルが存在するか確認
                start_sec, end_sec = self.calc_start_end_sec(ann_sec, self.sampling_sec)
                numpy_file = f"{self.feature_root}/{self.image_set}/{vid}_{start_sec}_{end_sec}_{ann_sec}_resnet.npy"
                if not os.path.isfile(numpy_file):
                    logger.warning(f"{numpy_file} doesn't exist")
                    continue
                self.ann[vid][key]["ann_secs"].append(ann_sec)
                self.ann[vid][key]["sentence"] = tokenizer(caption)
                self.ann[vid][key]["v_sec"] = v_sec


        self.vids = []
        for vid in self.ann:
            if (duration_df["vid"] == vid).any():
                # 動画の総時間を取得
                v_sec = duration_df[duration_df["vid"] == vid]["v_sec"].iloc[0]
                # csv　fileとtsvファイルで動画の時間が異なっているものを除外
                _v_sec = [v["v_sec"] for v in self.ann[vid].values()][0]
                if v_sec_thresh <= abs(_v_sec  - v_sec):
                    logger.warning(f"{vid} exceed time threshold({v_sec_thresh}): {v_sec}, {_v_sec}")
                    continue
            else:
                logger.warning(f"{vid} doesn't exist in duration file")
                continue

            #mp4ファイルが存在するか確認
            movie_file = f"{self.train_config['movie_root']}/v_{vid}.mp4"
            if not os.path.isfile(movie_file):
                logger.warning(f"{vid} doesn't exist in movie directory")
                continue


            # for caption, values in self.ann[vid].items():
            captions = tuple(self.ann[vid].keys())
            for caption in captions:
                values = self.ann[vid][caption]
                ann_secs = values["ann_secs"]

                
                # gold secがおかしいデータは削除
                if 0 < sum([v_sec < gold_sec for gold_sec in ann_secs]):
                    logger.warning(f"{vid} gold sec is over than total sec. {v_sec}, {ann_secs}")
                    del self.ann[vid][caption]
                    continue

                # 各データのgold_labelを作成
                # for gold_sec in ann_secs:
                #     start_sec, end_sec = self.calc_start_end_sec(gold_sec, self.sampling_sec)
                #     # numpyファイルが存在するか
                #     file_name = f"{self.feature_root}/{self.image_set}/{vid}_{start_sec}_{end_sec}_{gold_sec}_resnet.npy"
                #     if os.path.isfile(file_name):
                #         # フレーム数を取得
                #         frame_cnt = np.load(file_name).shape[0]  # ここはなんとかしたい
                #     else:
                #         logger.warning(f"{vid} doesn't exist in npy feature directory")
                #         continue

                #     self.ann[vid][caption]["start_sec"] = start_sec
                #     self.ann[vid][caption]["end_sec"] = end_sec
                #     self.ann[vid][caption]["labels"].append(self.calc_label(self.sampling_sec, [gold_sec - start_sec], frame_cnt, 0))
            self.vids.append(vid)

        if not self.is_sampling:
            # 複数のキャプションをvideoごとにまとめていたデータをserializeする
            self.serialized_ann = []
            for vid in self.vids:
                for ann in self.ann[vid].values():
                    ann["vid"] = vid
                    self.serialized_ann.append(ann)

        logger.info(f"{len(self.vids)} data was loaded")

    @staticmethod
    def calc_start_end_sec(gold_sec, sampling_sec):
        start_sec = 0.0
        while start_sec < gold_sec:
            start_sec += sampling_sec
        return start_sec - sampling_sec, start_sec

    @classmethod
    def get_submovie_tensor(cls, movie_file, gold_sec, start_sec, end_sec, model_names=["resnet", "bninception"]):
        ret = dict()
        for model_name in model_names:
            middle_feature = cls.vc(movie_file, model_name, sampling_sec=0.000001, start_sec=start_sec, end_sec=end_sec)
            v_sec = end_sec - start_sec
            ann_secs = [gold_sec - start_sec]
            gold_label, start_idx, end_idx = cls.calc_label(v_sec, ann_secs, middle_feature.size(0), 0)
            assert gold_label.size(0) == middle_feature.size(0)
            ret[model_name] = {"features": middle_feature.clone(), "label":gold_label.clone()}
        return ret, start_idx, end_idx


    def __len__(self):
        if self.is_sampling:
            return len(self.vids)
        else:
            return len(self.serialized_ann)


    def __getitem__(self, idx):
        if self.is_sampling:
            vid = self.vids[idx]
            ann = self.sampling(vid)
            sentence, ann_secs = ann["sentence"], ann["ann_secs"]
            # segmentのstartとend
            v_sec = ann["v_sec"]
            labels = ann["labels"]
        else:
            vid = self.serialized_ann[idx]["vid"]
            sentence = self.serialized_ann[idx]["sentence"]
            ann_secs = self.serialized_ann[idx]["ann_secs"]
            v_sec = self.serialized_ann[idx]["v_sec"]
            labels = self.serialized_ann[idx]["labels"]

        # 各キャプションのstart idxとend idx を取得できるように変更
        gold_index = random.randint(0, len(ann_secs)-1)
        gold_sec = ann_secs[gold_index]
        # gold_label = labels[gold_index]

        start_sec, end_sec = self.calc_start_end_sec(gold_sec, self.sampling_sec)
        
        resnet_feat = torch.from_numpy(
            np.load(f"{self.feature_root}/{self.image_set}/{vid}_{start_sec}_{end_sec}_{gold_sec}_resnet.npy")).float()
        bni_feat = torch.from_numpy(
            np.load(f"{self.feature_root}/{self.image_set}/{vid}_{start_sec}_{end_sec}_{gold_sec}_bn.npy")).float()
        
        # 毎回gold_labelを作成する
        gold_label, *_ = self.calc_label(self.sampling_sec, [gold_sec - start_sec], bni_feat.shape[0], 0)

        # ret, *_ = self.get_submovie_tensor(movie_file, gold_sec, start_sec, end_sec)
        # resnet_feat = ret["resnet"]["features"]
        # bni_feat = ret["bninception"]["features"]
        # gold_label = ret["resnet"]["label"]
        frame_cnt = bni_feat.size(0)
        common_label = gold_label.clone()

        sentence_len = len(sentence)

        assert gold_label.size() == common_label.size(), f"{gold_label.size()}, {common_label.size()}"

        return resnet_feat, bni_feat, sentence, frame_cnt, sentence_len,  gold_label, vid, common_label, None, None
