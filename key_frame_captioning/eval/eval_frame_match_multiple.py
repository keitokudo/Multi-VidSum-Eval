"""
    再起関数を用いた関数の実装
"""
import copy
from typing import List

from torch import Tensor
import numpy as np
from key_frame_captioning.eval.eval_frame_match import eval_func


from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

cos_sim_matrix = []  # pred num × gold numのマトリックス
pred_gold_index_matrix = []  # 引数としてもらった_pred_indexや_frame_data_allのindexとcos_sim_matrixのindexを対応つける
pred_index = []  # _pred_indexをglobalに保存
gold_index = []  # _frame_data_allのindexバージョン
upper_score = -float('inf')  #現在の上限値
pred_captions = []
gold_captions = []
upper_cos_sim_matrix = None


def reset_global_state():
    global pred_index
    global gold_index
    global pred_captions
    global gold_captions
    global upper_score
    global cos_sim_matrix
    global sim_matrix
    global pred_gold_index_matrix
    global upper_cos_sim_matrix
    cos_sim_matrix = []
    sim_matrix = []
    pred_gold_index_matrix = []
    pred_index = []
    gold_index = []
    upper_score = -float('inf')
    pred_captions = []
    gold_captions = []
    upper_cos_sim_matrix = None




def eval_frame_match_multiple(_pred_index: List[int], vis_feat: Tensor, _frame_data_all: Tensor, _pred_captions, _gold_captions, alpha=1):
    """
    コサイン類似度が最も高くなるようなpredとgoldの組み合わせを考える
    """
    # predとgoldの組み合わせ分のコサイン類似度を計算する
    # この時各goldには10人程度のアノテータがいるが，一旦時間は気にしないで最もコサイン類似度が高くなるようなアノテーションを使用する
    # print(len(_pred_index), vis_feat.shape, _frame_data_all.shape, len(_captions))
    global pred_index
    global gold_index
    global pred_captions
    global gold_captions
    global upper_score
    global cos_sim_matrix
    global sim_matrix
    global pred_gold_index_matrix
    global upper_cos_sim_matrix

    reset_global_state()

    pred_index = _pred_index
    gold_index = [[] for _ in range(len(_pred_index))]  # frmae_data_allをindexに直してる(複数人のアノーテータがいる中で一番コサイン類似度が高くなる人のindexを格納). pi, giを使った時はどのアノテータを使ってるのかを保存してる
    pred_captions = _pred_captions
    gold_captions = _gold_captions
    
    try:
        has_been_used = [False for _ in range(len(_frame_data_all))]
        cos_sim_matrix = [[0 for _ in range(len(_frame_data_all))] for _ in range(len(pred_index))]  # (gold size × pred size)
        sim_matrix = [[0 for _ in range(len(_frame_data_all))] for _ in range(len(pred_index))]  # (gold size × pred size)

        # predとgoldの組み合わせのコサイン類似度マトリックスを作成
        # 各goldに対して一番コサイン類似度が高くなるアノテータのindexを使用する
        assert len(_frame_data_all) == len(gold_captions)
        for pi in range(len(pred_index)):
            #preds = [pred_captions[pi]] * len(_frame_data_all)
            #golds = gold_captions
            #cap_scores = scorer.score(references=golds, candidates=preds)
            for gi in range(len(_frame_data_all)):
                _, _g_comb, _sim_value = eval_func([pred_index[pi]], vis_feat, _frame_data_all[gi].reshape(1, -1))

                cap_score = sentence_bleu([gold_captions[gi].split()], pred_captions[pi].split(), smoothing_function=SmoothingFunction().method1)

                #cos_sim_matrix[pi][gi] = ((1 + _sim_value) / 2 * alpha) + (cap_scores[gi] * (1 - alpha))
                cos_sim_matrix[pi][gi] = (_sim_value * alpha) + (cap_score * (1 - alpha))
                sim_matrix[pi][gi] = _sim_value

                # bleu scoreではセグメント中ではキャプションが全部一緒なのでフレームのindexは考慮しない
                gold_index[pi].append(_g_comb[0])


    # assert cos_sim.shape[0] == cap_score.shape[0], f"{cos_sim}, {cap_score}"

    # total_score = cos_sim * alpha + cap_score * (1-alpha)



        # predごとにコサイン類似度が高い順番でソートする
        cos_sim_matrix = np.array(cos_sim_matrix)
        sim_matrix = np.array(sim_matrix)

        pred_gold_index_matrix = np.argsort(-1 * cos_sim_matrix, axis=-1)
        # cos_sim_matrix = -1 * np.sort(-1*cos_sim_matrix, axis=-1)
        upper_cos_sim_matrix = (-1 * np.sort(-1*cos_sim_matrix, axis=-1))[:,0].reshape(-1)

        # print(gold_index)
        # print(cos_sim_matrix)

        # predAから順番に組み合わせを決定していく
        p_comb, g_comb, sim_value, c_score, selected_pred_captions, selected_gold_captions = dps(0, 0, [], [],[], [], len(gold_index), copy.deepcopy(has_been_used))
        return p_comb, g_comb, c_score, selected_pred_captions, selected_gold_captions
    except:
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


'''
再帰関数を用いてpredとgoldのマッチングを計算する関数
今まで一番高いスコアを用いて枝刈りをしている
'''
def dps(total_sim_value: float, total_score: float, prev_p_comb: List[int], prev_g_comb: List[int], prev_pred_captions: List[str], prev_gold_captions: List[str], max_select_num: int, has_been_used: List):
    """

    """
    global upper_score
    global cos_sim_matrix
    global sim_matrix
    global gold_index
    global pred_index
    global pred_captions
    global gold_captions
    global upper_cos_sim_matrix

    select_cnt = len(prev_p_comb)
    assert len(prev_p_comb) == sum(has_been_used), f"has_been_used:{has_been_used}, select_count:{select_cnt}"

    if select_cnt == max_select_num:
        # upper_scoreを最大値で更新
        if upper_score < total_sim_value:
            upper_score = total_sim_value
        return prev_p_comb, prev_g_comb, total_sim_value/max_select_num, total_score/max_select_num , prev_pred_captions, prev_gold_captions


    max_score = sum(upper_cos_sim_matrix[select_cnt:]) + total_sim_value   # 現在の状況から到達しうる最も高いスコア
    if max_score < upper_score:
        return None, None, -float('inf'), -float('inf'), None, None
    else:
        pi = select_cnt
        sim_value = -float('inf')
        c_score = None
        p_comb = None
        g_comb = None
        selected_pred_captions = None
        selected_gold_captions = None

        # コサイン類似度の組み合わせが高いものから見ていく
        # for gi in range(len(gold_index[pi])):
        for gi in pred_gold_index_matrix[pi]:
            #既に使用済みの場合は飛ばす
            if has_been_used[gi] is True:
                continue
            #次の再帰の引数の準備
            tmp_total_sim_value = total_sim_value + cos_sim_matrix[pi][gi]
            tmp_score = total_score + sim_matrix[pi][gi]
            _prev_p_comb = copy.deepcopy(prev_p_comb) + [pred_index[pi]]
            _prev_g_comb = copy.deepcopy(prev_g_comb) + [gold_index[pi][gi]]
            _pred_captions = copy.deepcopy(prev_pred_captions) + [pred_captions[pi]]
            _gold_captions = copy.deepcopy(prev_gold_captions) + [gold_captions[gi]]
            _has_been_used = copy.deepcopy(has_been_used)
            _has_been_used[gi] = True

            _p_comb, _g_comb, _sim_value, _c_score, _selected_pred_captions, _selected_gold_captions = dps(tmp_total_sim_value, tmp_score, _prev_p_comb, _prev_g_comb, _pred_captions, _gold_captions, max_select_num, _has_been_used)

            # sim_valueが最大の場合更新
            if sim_value < _sim_value:
                sim_value = _sim_value
                c_score = _c_score
                p_comb = _p_comb
                g_comb = _g_comb
                selected_pred_captions = _selected_pred_captions
                selected_gold_captions = _selected_gold_captions

        return p_comb, g_comb, sim_value, c_score, selected_pred_captions, selected_gold_captions


