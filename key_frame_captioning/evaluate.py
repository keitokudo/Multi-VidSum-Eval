import click
import yaml
import os
from pathlib import Path

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
from FrameDetectDataset import  FrameANetDataset
from tokenizer import get_tokenizer
from models.baseline import BaselineModel
from evaluator import Evaluator
from criterion import get_criterion

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
@click.option('--model_config_file', type=click.Path(exists=True), help="modelのパラメータなどの設定")
@click.option('--eval_config_file', type=click.Path(exists=True), help="config file")
@click.option('--model_dir', type=click.Path(exists=True), help="modelを保存しているディレクトリ")
@click.option('--res_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
@click.option("--debug", type=bool, is_flag=True, default=False, help="デバッグモードにする")
@click.option('--gpu', type=bool, is_flag=True, default=False, help="gpuを使うかどうか")
@click.option("--num_workers", type=int, default=6, help="前処理のcpu使用数")
@click.option("--max_captions", type=int, default=4, help="評価に使用する動画の最大キャプション数")
@click.option("--select_captions", type=int, default=100000, help="評価に使用する動画の最大キャプション数")
@click.option("--output_file_name", type=str, default="eval_frame_match_using_val1val2", help="出力ファイル名")
@click.option("--random_frame_sampling", type=bool, is_flag=True, default=False, help="フレーム抽出をランダムに行うベースラインの評価を行う")
@click.option("--feature",  type=click.Choice(['resnet', 'bni'], case_sensitive=False), default="resnet", help="フレームの比較に使用する特徴量")
@click.option("--use_sub_movie", type=bool, is_flag=True, default=False, help="動画を該当箇所のみ切り抜いて使用する")
@click.option('--dense_cap_file', type=click.Path(exists=True), default=None, help="densecapの出力jsonファイル")
@click.option("--seed", type=int, default=1)
@click.option('--use_only_movie_exist_vid', type=bool, is_flag=True, default=False, help="動画ファイルに含まれるもののみを使うかどうか")
@click.option("--alpha", type=float, default=1.0, help="コサイン類似度とbleuスコアとの比率")
def evaluate_frame_match_using_val1val2(model_config_file, model_dir, eval_config_file, res_dir, debug, gpu, num_workers, max_captions,select_captions, output_file_name, random_frame_sampling, feature, use_sub_movie, dense_cap_file, seed, use_only_movie_exist_vid, alpha):
    '''
    フレーム提示の性能を評価する
    '''
    image_set = "validation_testing"
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
            logger.warning("use_sub_movie was set True")
    if use_sub_movie is True:
        output_file_name = output_file_name + "_sub_movie"
    output_file_name += f"_alpha_{alpha}"
    output_file_name = output_file_name + f"_{feature}"
    if debug is True:
        output_file_name = output_file_name + "_debug"
    output_file_name += f"_{seed}"

    eval_config = yaml.safe_load(open(eval_config_file))
    model_config = yaml.safe_load(open(model_config_file))

    set_log_level(debug)
    log_file_path = Path(
        f"{res_dir}/{eval_config['save_dir']['log_dir']}/eval_frame_match_log.txt"
    )
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logzero.logfile(log_file_path)
    
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
    if model_dir is None:
        model.load_state_dict(
            torch.load(
                f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_{eval_config['use_model']}.pt"
            )["model"]
        )
    else:
        model.load_state_dict(
            torch.load(
                f"{model_dir}/model_{eval_config['use_model']}.pt"
            )["model"]
        )
    model.eval()

    
    # dev_test_dataset = ANetDataset(
    #     eval_config, tokenizer, pad_token_id, vocab, debug, image_set=image_set, use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file, use_only_movie_exist_vid=use_only_movie_exist_vid)
    dev_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="validation", use_sub_movie=use_sub_movie, use_only_movie_exist_vid=use_only_movie_exist_vid)
    test_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="testing", use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file, use_only_movie_exist_vid=use_only_movie_exist_vid)

    evaluator = Evaluator(model, device, res_dir, eval_config,
                            vocab, "validation", debug=debug)

    dev_loader = DataLoader(dev_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=dev_dataset.collate_fn)
    test_loader = DataLoader(test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=test_dataset.collate_fn)


    evaluator.eval_frame_match_using_val1val2(dev_loader, test_loader, max_captions, select_captions, output_file_name, feature, is_random=random_frame_sampling, use_sub_movie=use_sub_movie, alpha=alpha)


@cli.command()
@click.option('--model_config_file', type=click.Path(exists=True), help="modelのパラメータなどの設定")
@click.option('--eval_config_file', type=click.Path(exists=True), help="config file")
@click.option('--res_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
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
def evaluate_frame_match_using_val1val2_selected_numbers(model_config_file, eval_config_file, res_dir, debug, gpu, num_workers, max_captions, output_file_name, random_frame_sampling, feature, use_sub_movie, dense_cap_file, seed, use_only_movie_exist_vid, alpha):
    '''
    フレーム提示の性能を評価する
    '''
    image_set = "validation_testing"
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
    output_file_name += f"_alpha_{alpha}"
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
    model.load_state_dict(torch.load(f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_{eval_config['use_model']}.pt")["model"])
    model.eval()

    
    dev_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="validation", use_sub_movie=use_sub_movie, use_only_movie_exist_vid=use_only_movie_exist_vid)
    selected_test_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="validation_testing", use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file, use_only_movie_exist_vid=use_only_movie_exist_vid)
    oracle_test_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="testing", use_sub_movie=use_sub_movie, use_only_movie_exist_vid=use_only_movie_exist_vid)        

    evaluator = Evaluator(model, device, res_dir, eval_config,
                            vocab, "validation", debug=debug)

    dev_loader = DataLoader(dev_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=dev_dataset.collate_fn)
    selected_test_loader = DataLoader(selected_test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=selected_test_dataset.collate_fn)
    oracle_test_loader = DataLoader(oracle_test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=oracle_test_dataset.collate_fn)                             


    evaluator.eval_frame_match_using_val1val2_by_selected_numbers(dev_loader, oracle_test_loader, selected_test_loader, max_captions, output_file_name, feature, is_random=random_frame_sampling, use_sub_movie=use_sub_movie,alpha=alpha)


@cli.command()
@click.option('--model_config_file', type=click.Path(exists=True), help="modelのパラメータなどの設定")
@click.option('--eval_config_file', type=click.Path(exists=True), help="config file")
@click.option('--res_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
@click.option("--debug", type=bool, is_flag=True, default=False, help="デバッグモードにする")
@click.option('--gpu', type=bool, is_flag=True, default=False, help="gpuを使うかどうか")
@click.option("--num_workers", type=int, default=6, help="前処理のcpu使用数")
@click.option("--max_captions", type=int, default=6, help="評価に使用する動画の最大キャプション数")
@click.option("--output_file_name", type=str, default="eval_frame_match", help="出力ファイル名")
@click.option("--random_frame_sampling", type=bool, is_flag=True, default=False, help="フレーム抽出をランダムに行うベースラインの評価を行う")
@click.option("--feature",  type=click.Choice(['resnet', 'bni'], case_sensitive=False), default="resnet", help="フレームの比較に使用する特徴量")
@click.option("--use_sub_movie", type=bool, is_flag=True, default=False, help="動画を該当箇所のみ切り抜いて使用する")
@click.option("--image_set", type=click.Choice(['training', 'validation', 'testing']), default="testing",  help="使用するデータ")
@click.option('--dense_cap_file', type=click.Path(exists=True), default=None, help="densecapの出力jsonファイル")
@click.option("--seed", type=int, default=1)
@click.option('--use_only_movie_exist_vid', type=bool, is_flag=True, default=False, help="動画ファイルに含まれるもののみを使うかどうか")
def evaluate_frame_match(model_config_file, eval_config_file, res_dir, debug, gpu, num_workers, max_captions, output_file_name, random_frame_sampling, feature, use_sub_movie, image_set, dense_cap_file, seed, use_only_movie_exist_vid):
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
    model.load_state_dict(torch.load(f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_{eval_config['use_model']}.pt")["model"])
    model.eval()

    test_dataset = ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set=image_set, use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file, use_only_movie_exist_vid=use_only_movie_exist_vid)
    mode = "validation" if image_set == "testing" else image_set
    evaluator = Evaluator(model, device, res_dir, eval_config,
                            vocab, mode, debug=debug)

    test_loader = DataLoader(test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=test_dataset.collate_fn)

    evaluator.eval_frame_match(test_loader, max_captions, output_file_name, feature, is_random=random_frame_sampling, use_sub_movie=use_sub_movie)


@cli.command()
@click.option('--segment_model_config_file', type=click.Path(exists=True), help="segment detect modelのパラメータなどの設定")
@click.option('--frame_model_config_file', type=click.Path(exists=True), help="frame detect modelのパラメータなどの設定")
@click.option('--segment_eval_config_file', type=click.Path(exists=True), help="segment config file")
@click.option('--frame_eval_config_file', type=click.Path(exists=True), help="frame config file")
@click.option('--segment_res_dir', type=click.Path(exists=False), help="segment detect modelのディレクトリ")
@click.option('--frame_res_dir', type=click.Path(exists=False), help="frame detect modelのディレクトリ")
@click.option("--debug", type=bool, is_flag=True, default=False, help="デバッグモードにする")
@click.option('--gpu', type=bool, is_flag=True, default=False, help="gpuを使うかどうか")
@click.option("--num_workers", type=int, default=6, help="前処理のcpu使用数")
@click.option("--max_captions", type=int, default=6, help="評価に使用する動画の最大キャプション数")
@click.option("--output_file_name", type=str, default="eval_frame_match", help="出力ファイル名")
@click.option("--random_frame_sampling", type=bool, is_flag=True, default=False, help="フレーム抽出をランダムに行うベースラインの評価を行う")
@click.option("--feature",  type=click.Choice(['resnet', 'bni'], case_sensitive=False), default="resnet", help="フレームの比較に使用する特徴量")
@click.option("--use_sub_movie", type=bool, is_flag=True, default=False, help="動画を該当箇所のみ切り抜いて使用する")
@click.option("--image_set", type=click.Choice(['training', 'validation', 'testing']), default="testing",  help="使用するデータ")
@click.option('--dense_cap_file', type=click.Path(exists=True), default=None, help="densecapの出力jsonファイル")
@click.option("--seed", type=int, default=1)
def evaluate_frame_match_with_two_model(segment_model_config_file, frame_model_config_file, segment_eval_config_file, frame_eval_config_file, segment_res_dir, frame_res_dir, debug, gpu, num_workers, max_captions, output_file_name, random_frame_sampling, feature, use_sub_movie, image_set, dense_cap_file, seed):
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

    segment_eval_config = yaml.safe_load(open(segment_eval_config_file))
    frame_eval_config = yaml.safe_load(open(frame_eval_config_file))
    segment_model_config = yaml.safe_load(open(segment_model_config_file))
    frame_model_config = yaml.safe_load(open(frame_model_config_file))

    set_log_level(debug)
    logzero.logfile(
        f"{frame_res_dir}/{frame_eval_config['save_dir']['log_dir']}/eval_frame_match_with_two_model_log.txt", 'w')

    # ディレクトリの作成
    for dir_name in frame_eval_config["save_dir"].values():
        os.makedirs(f"{frame_res_dir}/{dir_name}", exist_ok=True)
    if debug is False:
        assert not os.path.isfile(f"{frame_res_dir}/{frame_eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"), f"{frame_res_dir}/{frame_eval_config['save_dir']['eval_dir']}/{output_file_name}.pickle"

    logger.info(frame_eval_config)

    tokenizer, vocab, pad_token_id = get_tokenizer(
        frame_model_config["model"]["language_model"], frame_eval_config["dataset_file"])

    segment_model = BaselineModel(segment_model_config["model"], device)
    frame_model = BaselineModel(frame_model_config["model"], device)

    # モデルをload
    segment_model.load_state_dict(torch.load(f"{segment_res_dir}/{segment_eval_config['save_dir']['model_dir']}/model_{segment_eval_confi['use_model']}.pt")["model"])
    segment_model.eval()
    frame_model.load_state_dict(torch.load(f"{frame_res_dir}/{frame_eval_config['save_dir']['model_dir']}/model_{frame_eval_config['use_model']}.pt")["model"])
    frame_model.eval()

    test_dataset = ANetDataset(
        segment_eval_config, tokenizer, pad_token_id, vocab, debug, image_set=image_set, use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file)
    # frame_dataset = FrameANetDataset(frame_eval_config, tokenizer, pad_token_id, vocab, debug, image_set=image_set, dense_cap_file=dense_cap_file)
    mode = "validation" if image_set == "testing" else image_set
    evaluator = Evaluator(segment_model, device, frame_res_dir, segment_eval_config, vocab, mode, debug=debug, frame_model=frame_model)

    test_loader = DataLoader(test_dataset,
                             segment_eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=test_dataset.collate_fn)

    evaluator.eval_frame_match_with_two_model(test_loader, max_captions, output_file_name, feature, is_random=random_frame_sampling, use_sub_movie=use_sub_movie)



@cli.command()
@click.option('--model_config_file', type=click.Path(exists=True), help="modelのパラメータなどの設定")
@click.option('--eval_config_file', type=click.Path(exists=True), help="config file")
@click.option('--res_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
@click.option("--debug", type=bool, is_flag=True, default=False, help="デバッグモードにする")
@click.option('--gpu', type=bool, is_flag=True, default=False, help="gpuを使うかどうか")
@click.option("--num_workers", type=int, default=6, help="前処理のcpu使用数")
@click.option("--output_file_name", type=str, default="eval_acc_loss", help="出力ファイル名")
@click.option("--random_frame_sampling", type=bool, is_flag=True, default=False, help="フレーム抽出をランダムに行うベースラインの評価を行う")
@click.option("--use_sub_movie", type=bool, is_flag=True, default=False, help="動画を該当箇所のみ切り抜いて使用する")
@click.option("--image_set", type=click.Choice(['training', 'validation', 'testing']), default="testing",  help="使用するデータ")
@click.option('--dense_cap_file', type=click.Path(exists=True), default=None, help="densecapの出力jsonファイル")
def evaluate_acc_loss(model_config_file, eval_config_file, res_dir, debug, gpu, num_workers, output_file_name, random_frame_sampling, use_sub_movie, image_set, dense_cap_file):
    """
    accuracyとlossを計算する（accuracyといいつつ本当はprecision）
    """
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
    if debug is True:
        output_file_name = output_file_name + "_debug"

    eval_config = yaml.safe_load(open(eval_config_file))
    model_config = yaml.safe_load(open(model_config_file))

    set_log_level(debug)
    logzero.logfile(
        f"{res_dir}/{eval_config['save_dir']['log_dir']}/{output_file_name}_log.txt")

    # ディレクトリの作成
    for dir_name in eval_config["save_dir"].values():
        os.makedirs(f"{res_dir}/{dir_name}", exist_ok=True)
    assert not os.path.isfile(f"{res_dir}/{eval_config['save_dir']['eval_dir']}/{output_file_name}.json")

    device = "cuda" if torch.cuda.is_available() and gpu is True else "cpu"
    logger.info(f"use {device}")

    logger.info(eval_config)

    tokenizer, vocab, pad_token_id = get_tokenizer(
        model_config["model"]["language_model"], eval_config["dataset_file"])

    model = BaselineModel(model_config["model"], device)

    criterion = get_criterion(model_config["criterion"])
    logger.debug(criterion)

    # モデルをload
    model.load_state_dict(torch.load(f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_{eval_config['model']}.pt")["model"])
    model.eval()

    test_dataset = ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set=image_set, use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file)
    mode = "validation" if image_set == "testing" else image_set
    evaluator = Evaluator(model, device, res_dir, eval_config,
                            vocab, mode, debug=debug, criterion=criterion)

    test_loader = DataLoader(test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=test_dataset.collate_fn)

    evaluator.eval_acc_loss(test_loader, output_file_name, is_random=random_frame_sampling)



@cli.command()
@click.option('--model_config_file', type=click.Path(exists=True), help="modelのパラメータなどの設定")
@click.option('--eval_config_file', type=click.Path(exists=True), help="config file")
@click.option('--res_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
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
def evaluate_version3(model_config_file, eval_config_file, res_dir, debug, gpu, num_workers, max_captions, output_file_name, random_frame_sampling, feature, use_sub_movie, dense_cap_file, seed, use_only_movie_exist_vid, alpha, is_predict_frame, method, selected_alpha):
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

    # output_file_name = image_set + '_' + output_file_name
    # if random_frame_sampling is True:
    #     logger.info("random sampling mode")
    #     output_file_name = output_file_name + "_random_sampling"
    # if is_predict_frame is not True:
    #     output_file_name = output_file_name + "_not_predict"
    # if dense_cap_file is not None:
    #     output_file_name = output_file_name + "_densecap"
    #     if use_sub_movie is False:
    #         use_sub_movie = True
    #         logger.warning(f"use_sub_movie was set True")
    # if use_sub_movie is True:
    #     output_file_name = output_file_name + "_sub_movie"
    # output_file_name += f"_alpha_{alpha}"
    # output_file_name = output_file_name + f"_{feature}"
    # if debug is True:
    #     output_file_name = output_file_name + "_debug"
    # output_file_name += f"_{seed}"

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
    model.load_state_dict(torch.load(f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_{eval_config['use_model']}.pt")["model"])
    model.eval()



    oracle_dev_dataset = ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="validation", use_sub_movie=use_sub_movie, use_only_movie_exist_vid=use_only_movie_exist_vid)
    selected_test_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="validation_testing", use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file, use_only_movie_exist_vid=use_only_movie_exist_vid)
    oracle_test_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="testing", use_sub_movie=use_sub_movie, use_only_movie_exist_vid=use_only_movie_exist_vid)        

    evaluator = Evaluator(model, device, res_dir, eval_config,
                            vocab, "validation", debug=debug)

    dev_loader = DataLoader(oracle_dev_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=oracle_dev_dataset.collate_fn)
    selected_test_loader = DataLoader(selected_test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=selected_test_dataset.collate_fn)
    oracle_test_loader = DataLoader(oracle_test_dataset,
                             eval_config['batchsize'],
                             num_workers=num_workers,
                             shuffle=False,
                             collate_fn=oracle_test_dataset.collate_fn)                             

    # if is_cos_evaluate:
    evaluator.eval_frame_match_version3(dev_loader, oracle_test_loader, selected_test_loader, max_captions, output_file_name, feature, method, is_random=random_frame_sampling, use_sub_movie=use_sub_movie,alpha=alpha, is_predict_frame=is_predict_frame, selected_alpha=selected_alpha)
    # else:
    #     evaluator.predict_version3(dev_loader, oracle_test_loader, selected_test_loader, max_captions, output_file_name, feature, is_random=random_frame_sampling, use_sub_movie=use_sub_movie,alpha=alpha, is_predict_frame=is_predict_frame)

@cli.command()
@click.option('--model_config_file', type=click.Path(exists=True), help="modelのパラメータなどの設定")
@click.option('--eval_config_file', type=click.Path(exists=True), help="config file")
@click.option('--model_dir', type=click.Path(exists=True), help="modelを保存しているディレクトリ")
@click.option('--res_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
@click.option("--debug", type=bool, is_flag=True, default=False, help="デバッグモードにする")
@click.option('--gpu', type=bool, is_flag=True, default=False, help="gpuを使うかどうか")
@click.option("--num_workers", type=int, default=6, help="前処理のcpu使用数")
@click.option("--output_file_name", type=str, default="eval_frame_match_using_val1val2", help="出力ファイル名")
@click.option("--random_frame_sampling", type=bool, is_flag=True, default=False, help="フレーム抽出をランダムに行うベースラインの評価を行う")
@click.option("--use_sub_movie", type=bool, is_flag=True, default=False, help="動画を該当箇所のみ切り抜いて使用する")
@click.option('--dense_cap_file', type=click.Path(exists=True), default=None, help="densecapの出力jsonファイル")
@click.option("--seed", type=int, default=1)
@click.option('--use_only_movie_exist_vid', type=bool, is_flag=True, default=False, help="動画ファイルに含まれるもののみを使うかどうか")
@click.option("--is_predict_frame", type=bool, is_flag=True, default=False, help="モデルを用いて予測を行う")
def evaluate_version3_predict_only(model_config_file, eval_config_file, model_dir, res_dir, debug, gpu, num_workers, output_file_name, random_frame_sampling, use_sub_movie, dense_cap_file, seed, use_only_movie_exist_vid, is_predict_frame):
    '''
    pickleデータ(評価のためのval1とval2を合わせたgoldや予測したフレーム等の情報が記録されている)
    jsonデータ(次のroopに必要な予測したフレーム等の情報が記録されている）
    を作成する
    '''
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
    log_file_path = Path(
        f"{res_dir}/{eval_config['save_dir']['log_dir']}/eval_frame_match_log.txt"
    )
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logzero.logfile(log_file_path)
    
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
    if model_dir is None:
        model.load_state_dict(
            torch.load(
                f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_{eval_config['use_model']}.pt"
            )["model"]
        )
    else:
        model.load_state_dict(
            torch.load(
                f"{model_dir}/model_{eval_config['use_model']}.pt"
            )["model"]
        )
    model.eval()

    selected_test_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="validation_testing", use_sub_movie=use_sub_movie, dense_cap_file=dense_cap_file, use_only_movie_exist_vid=use_only_movie_exist_vid
    )
    oracle_test_dataset =  ANetDataset(
        eval_config, tokenizer, pad_token_id, vocab, debug, image_set="fixed_eval", use_sub_movie=use_sub_movie, use_only_movie_exist_vid=use_only_movie_exist_vid
    )

    evaluator = Evaluator(model, device, res_dir, eval_config,
                            vocab, "validation", debug=debug)

    selected_test_loader = DataLoader(
        selected_test_dataset,
        eval_config['batchsize'],
        num_workers=num_workers,
        shuffle=False,
        collate_fn=selected_test_dataset.collate_fn
    )
    oracle_test_loader = DataLoader(
        oracle_test_dataset,
        eval_config['batchsize'],
        num_workers=num_workers,
        shuffle=False,
        collate_fn=oracle_test_dataset.collate_fn
    )
    
    evaluator.eval_frame_match_version3_predict_only(oracle_test_loader, selected_test_loader, output_file_name, is_random=random_frame_sampling, use_sub_movie=use_sub_movie, is_predict_frame=is_predict_frame)


@cli.command()
@click.option('--model_config_file', type=click.Path(exists=True), help="modelのパラメータなどの設定")
@click.option('--eval_config_file', type=click.Path(exists=True), help="config file")
@click.option('--model_dir', type=click.Path(exists=True), help="modelを保存しているディレクトリ")
@click.option('--res_dir', type=click.Path(exists=False), help="結果を出力するディレクトリ")
@click.option("--debug", type=bool, is_flag=True, default=False, help="デバッグモードにする")
@click.option('--gpu', type=bool, is_flag=True, default=False, help="gpuを使うかどうか")
@click.option("--max_captions", type=int, default=100000000, help="評価に使用する動画の最大キャプション数")
@click.option("--output_file_name", type=str, default="eval_frame_match_using_val1val2", help="出力ファイル名")
@click.option("--feature",  type=click.Choice(['resnet', 'bni'], case_sensitive=False), default="resnet", help="フレームの比較に使用する特徴量")
@click.option("--seed", type=int, default=1)
@click.option("--alpha", type=float, default=1.0, help="コサイン類似度とbleuスコアとの比率")
@click.option("--method",  type=click.Choice(['greedy', 'dp', 'random'], case_sensitive=False), help="評価に使用するセグメントを選ぶ手法")
@click.option("--selected_alpha", type=float, help="dpやgreedyにおいてframe scoreとcaption scoreの割合")
@click.option('--pickle_path', type=click.Path(exists=True), help="predict onlyで生成されたpickle_path")
@click.option('--json_path', type=click.Path(exists=True), help="predict onlyで生成されたjson_path")
def evaluate_version3_eval_only(model_config_file, eval_config_file, model_dir, res_dir, debug, gpu, max_captions, output_file_name, feature, seed, alpha, method, selected_alpha, pickle_path, json_path):
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

    eval_config = yaml.safe_load(open(eval_config_file))
    model_config = yaml.safe_load(open(model_config_file))

    set_log_level(debug)
    log_file_path = Path(
        f"{res_dir}/{eval_config['save_dir']['log_dir']}/eval_frame_match_log.txt"
    )
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logzero.logfile(log_file_path)
    
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
    if model_dir is None:
        model.load_state_dict(
            torch.load(
                f"{res_dir}/{eval_config['save_dir']['model_dir']}/model_{eval_config['use_model']}.pt"
            )["model"]
        )
    else:
        model.load_state_dict(
            torch.load(
                f"{model_dir}/model_{eval_config['use_model']}.pt"
            )["model"]
        )
    model.eval()

    

    
    evaluator = Evaluator(model, device, res_dir, eval_config,
                            vocab, "validation", debug=debug)
                          

    evaluator.eval_frame_match_version3_eval_only(max_captions, output_file_name, feature, method, pickle_path, json_path, alpha=alpha, selected_alpha=selected_alpha)


if __name__ == "__main__":
    cli()
