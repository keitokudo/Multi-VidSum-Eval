#!/bin/bash
### jsonを受け取って評価するスクリプト

seed=0
pred=${1} # prediction file (json)
cap=${2} # caption model [fairseq, clip_prefix, clip_reward, instruct_blip_t5_zero_shot, instruct_blip_t5_few_shot, vid2seq, self]
TAG=${3} # tag for output dir
NUM_KEY_FRAMES_AND_CAPTIONS=4

# Setting
source setting.sh
set -ex
export PYTHONPATH="$GITHUB_DIR:$PYTHONPATH"
MODEL_CONFIG=${GITHUB_DIR}/config/EMNLP2023/model_bilstm_config.yml
EVAL_CONFIG=${GITHUB_DIR}/config/EMNLP2023/train_bilstm_config.yml

OUTPUT_PATH=${GITHUB_DIR}/outputs/${TAG}

mkdir -p $OUTPUT_PATH

### matching pred and gold
python ${GITHUB_DIR}/key_frame_captioning/eval_reranking_model.py evaluate-version3 \
    --pred ${pred}\
    --cap ${cap}\
    --output_dir ${OUTPUT_PATH} \
    --output_file_name eval_frame_match_fixed_eval \
    --debug \
    --model_config_file ${MODEL_CONFIG} \
    --eval_config_file ${EVAL_CONFIG} \
    --res_dir ${RES_DIR} \
    --gpu --use_sub_movie --seed ${seed} \
    --max_captions 4 \
    --is_predict_frame --method dp --alpha 1.0 \
    --preprocessed_caption_dir $PREPROCESSED_CAPTION_DIR


PICKLE_PATH=${OUTPUT_PATH}/eval_frame_match_fixed_eval

### extract pred.txt and gold.txt from pickle file
python3 ${GITHUB_DIR}/preprocess/alignment_densecap_and_gold_caption.py \
	-p ${PICKLE_PATH}.pickle \
	-o ${OUTPUT_PATH}



# Add period to pred etc.
cp ${OUTPUT_PATH}/gold.txt ${OUTPUT_PATH}/gold_normalized.txt
cat ${OUTPUT_PATH}/pred.txt | python ${GITHUB_DIR}/postprocess/add_period.py > ${OUTPUT_PATH}/pred_normalized.txt


### preprocess before evaluation
### lowercase and tokenize, add period to pred
cat ${OUTPUT_PATH}/gold_normalized.txt | ${MOSES_DIR}/scripts/tokenizer/lowercase.perl | ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -no-escape > ${OUTPUT_PATH}/gold.tkn
cat ${OUTPUT_PATH}/pred_normalized.txt | ${MOSES_DIR}/scripts/tokenizer/lowercase.perl | ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -no-escape > ${OUTPUT_PATH}/pred.tkn


## compute meteor score
java -Xmx2G -jar ${METEOR_DIR}/meteor-*.jar ${OUTPUT_PATH}/pred.tkn ${OUTPUT_PATH}/gold.tkn -l en -norm > ${OUTPUT_PATH}/meteor.score


## compute bleurt score
python -m bleurt.score_files -candidate_file=${OUTPUT_PATH}/pred_normalized.txt -reference_file=${OUTPUT_PATH}/gold_normalized.txt -bleurt_checkpoint=${BLEURT_DIR}/bleurt/BLEURT-20 -scores_file=${OUTPUT_PATH}/bleurt.score

### compute akm score from pickle file
python3 ${GITHUB_DIR}/key_frame_captioning/analysis/eval_frame_match.py -i ${PICKLE_PATH}.pickle > ${OUTPUT_PATH}/akm.score

### compute score of concat captions
bash eval_concat.sh ${OUTPUT_PATH} ${NUM_KEY_FRAMES_AND_CAPTIONS}
