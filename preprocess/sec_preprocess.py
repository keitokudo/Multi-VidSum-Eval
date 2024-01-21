import json
import os

import numpy as np

class Preprocess:
    def __init__(self):
        duration_path = "/work/downloads/data/anet_duration_frame.csv"
        train_val_path = "/work/downloads/data/anet_annotations_trainval.json"
        self.numpy_dir_path = "/work/downloads/data/features"
        
        self.duration_data = dict()
        with open(duration_path) as f:
            for line in f:
                vid, movie_sec, _ = line.rstrip().split(',')
                self.duration_data[vid] = float(movie_sec)

        self.train_val_data = json.load(open(train_val_path))["database"]
        
    def get_numpy_path(self, vid):
        vid = self.cut_vid(vid)
        for mode in ["fixed_eval"]:
            numpy_path = f"{self.numpy_dir_path}/{mode}/{vid}_bn.npy"
            if os.path.isfile(numpy_path):
                return numpy_path
        return None

    def get_frame_cnt(self, vid):
        numpy_path = self.get_numpy_path(vid)
        if numpy_path is None:
            return None
        else:
            return np.load(numpy_path).shape[0]

    def trainval2duration_sec(self, vid, trainval_sec):
        vid = self.cut_vid(vid)
        if vid not in self.duration_data or vid not in self.train_val_data:
            return None
        return min(trainval_sec * self.duration_data[vid]/self.train_val_data[vid]["duration"], self.duration_data[vid])
    
    def duration_sec2trainvalsec(self, vid, duration_sec):
        vid = self.cut_vid(vid)
        return min(duration_sec * self.train_val_data[vid]["duration"]/self.duration_data[vid], self.train_val_data[vid]["duration"])

    def sec2index(self, vid, duration_sec):
        '''duration secに合わせてindexに変換する'''
        vid = self.cut_vid(vid)
        v_sec = self.duration_data[vid]
        frame_cnt = self.get_frame_cnt(vid)
        if v_sec is None or frame_cnt is None:
            return None
        frame_per_sec = v_sec / (frame_cnt - 1)
        return int((duration_sec + (frame_per_sec/2)) / frame_per_sec)

    def index2sec(self, vid, index):
        '''duration secに直す．trainval_secが欲しい時はduration_sec2trainval_secを使用する'''
        vid = self.cut_vid(vid)
        v_sec = self.duration_data[vid]
        frame_cnt = self.get_frame_cnt(vid)
        frame_per_sec = v_sec / (frame_cnt - 1)
        return  index * frame_per_sec

    @staticmethod
    def cut_vid(vid):
        if len(vid) == 13:
            vid = vid[2:]
        return vid
