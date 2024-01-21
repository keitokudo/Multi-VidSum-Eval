import sys

from pandas.core import frame
sys.path.append("../..")

from key_frame_captioning.eval.eval_frame_match_multiple import eval_frame_match_multiple
import numpy as np

def test():
    pred_index = [1, 1, 2, 3]
    captions = list(map(str, pred_index))
    vis_feat = np.random.rand(10, 100)
    frame_data_all = np.array([[0 for _ in range(10)] for _ in range(6)])
    for i in range(len(frame_data_all)):
        for _ in range(2):
            index = np.random.randint(0, 9, (2))
            frame_data_all[i, index] = 1


    p_comb, g_comb, sim_value, selected_captions = eval_frame_match_multiple(pred_index, vis_feat, frame_data_all, captions)

    print(p_comb, g_comb, sim_value, selected_captions)


if __name__ == "__main__":
    test()
