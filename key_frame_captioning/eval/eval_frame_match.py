# video　z6g5QbIPatkをいじる
# 入力
# 1 予測のindex  n個要素のリスト
# 2 resnetの情報
# 3 Bn_inceptionの情報
# 4 各キャプションに関する北山キャプションのフレーム情報 nセット

# 出力 類似度が最大に一番似てるidの組み合わせとその値

import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity


def eval_func(pred_index, vid_feat, frame_data_all):

    n = (len(frame_data_all))

    # 画像情報の選択　今回はResnetの特徴量だけで類似度計算する
    # bnを使う選択肢もあるし，resnetとbnの情報をフレームの数が変わらないように混ぜてもいいとは思う
    # vis_data  (frame_cnt,vis_dim)

    # 北山キャプションに該当するframeの画像特徴量の抽出

    # 1動画の中で北山キャプションが最もついている数(max_num)を抽出して(北山キャプション数(n),max_num, 画像特徴量の次元数)
    # のデータを作る．これが予測された箇所の画像特徴量と類似度計算される行列(gold_vid_feat)になる
    max_num = np.max(np.sum(frame_data_all, 1))
    gold_vid_feat = np.zeros((n, int(max_num), vid_feat.shape[1]))

    gold_vid_feat_frame_id = np.zeros((n, int(max_num), 1), dtype=int)

    block_id = 0
    cap_flag = 0
    for cap_id, frame_num in zip(np.where(frame_data_all == 1)[0], np.where(frame_data_all == 1)[1]):
        if cap_flag != cap_id:
            block_id = 0
            cap_flag = cap_id

        gold_vid_feat[cap_id][block_id] = vid_feat[frame_num]
        # argmax用にframeを入れる
        gold_vid_feat_frame_id[cap_id][block_id] = frame_num
        block_id += 1

#     print(gold_vid_feat_frame_id)

    # 全組み合わせを計算

    # predの全組み合わせについて予測された画像特徴量とgold_vid_featを「元々のdenseキャプションごとのblockごとに」類似度計算して平均する
    # 類似度の最大値が更新されるとその値sim_max_valueとその時のpredのidの組み合わせ(max_comb)が更新される

    sim_max_value = -1
    for comb in itertools.permutations(pred_index):
        sim_total = 0
        gold_comb = []

        for block_id in range(n):

            # このsim_argにgoldで一番類似度が大きいframeのid
            sim_arg = np.argmax(cosine_similarity(
                gold_vid_feat[block_id], vid_feat[comb[block_id]].reshape(1, -1)))
#             print(sim_arg)
            sim_total += cosine_similarity(gold_vid_feat[block_id][sim_arg].reshape(
                1, -1), vid_feat[comb[block_id]].reshape(1, -1))[0][0]
            gold_comb.append(int(gold_vid_feat_frame_id[block_id][sim_arg]))

        sim_total = sim_total / n


        if sim_total > sim_max_value:
            sim_max_value = sim_total
            max_comb = comb
            gold_max_comb = gold_comb

    return np.array(max_comb), np.array(gold_max_comb), sim_max_value


