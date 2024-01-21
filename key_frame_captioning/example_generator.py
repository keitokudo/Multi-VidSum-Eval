import click
import yaml
import os

# logger
import logzero
from logzero import logger
import logging

import random
import numpy as np

# torch
import torch
from torch.utils.data import DataLoader

# our implements
from Dataset import ANetDataset
from tokenizer import get_tokenizer
from models.baseline import BaselineModel
from evaluator import Evaluator
from criterion import get_criterion

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocess.video_preprocess import VideoConverter

vc = VideoConverter("cuda")


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


@cli.command()
@click.option("--debug", type=bool, is_flag=True, default=False, help="デバッグモードにする")
@click.option('--gpu', type=bool, is_flag=True, default=False, help="gpuを使うかどうか")
@click.option("--num_workers", type=int, default=6, help="前処理のcpu使用数")
@click.option("--frame_match_pickle", type=click.Path(exists=True), help="frame match の結果pickle file")
def example_generator(debug, gpu, num_workers, frame_match_pickle):
    '''
    フレーム提示の性能を評価する
    '''
    device = "cuda" if torch.cuda.is_available() and gpu is True else "cpu"
    logger.info(f"use {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    output_file_name = image_set + '_' + output_file_name
    if random_frame_sampling is True:
        logger.info("random sampling mode")
        output_file_name = output_file_name + "_random_sampling"
    if dense_cap_file is not None:
        output_file_name = output_file_name + "_densecap"
        if use_sub_movie is False:
            use_sub_movie = True
            logger.warning(f"use_sub_movie was set True")
    if use_sub_movie is True:
        output_file_name = output_file_name + "_sub_movie"
    output_file_name = output_file_name + f"_{feature}"
    if debug is True:
        output_file_name = output_file_name + "_debug"
    output_file_name += f"_{seed}"

    eval_config = yaml.safe_load(open(eval_config_file))
    model_config = yaml.safe_load(open(model_config_file))

    set_log_level(debug)
    logzero.logfile(
        f"{res_dir}/{eval_config['save_dir']['log_dir']}/eval_frame_match_log.txt")

    # ディレクトリの作成
    for dir_name in eval_config["save_dir"].values():
        os.makedirs(f"{res_dir}/{dir_name}", exist_ok=True)
    if debug is False:
        assert not os.path.isfile(f"{res_dir}/{eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"), f"{res_dir}/{eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"


    logger.info(eval_config)

    tokenizer, vocab, pad_token_id = get_tokenizer(
        model_config["model"]["language_model"], eval_config["dataset_file"])

    model = BaselineModel(model_config["model"], device)

    # モデルをload
    model.load_state_dict(torch.load(f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_best.pt")["model"])
    model.eval()

    test_dataset = ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set=image_set, use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file)
    mode = "validation" if image_set == "testing" else image_set
    evaluator = Evaluator(model, device, res_dir, eval_config,
                            vocab, mode, debug=debug)

    test_loader = DataLoader(test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=test_dataset.collate_fn)

    evaluator.eval_frame_match(test_loader, max_captions, output_file_name, feature, is_random=random_frame_sampling, use_sub_movie=use_sub_movie)


def create_key_frame(video_path:str, sec:float, output_path:str):
    image = vc.convert_video_to_image(video_path, True, 0.00001, sec, sec)
    image = image[0]
    image.save(output_path)



if __name__ == "__main__":
    cli()