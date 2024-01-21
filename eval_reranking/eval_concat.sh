source setting.sh
export PYTHONPATH="$GITHUB_DIR:$PYTHONPATH"
OUTPUT_PATH=$1
NUM_KEY_FRAMES_AND_CAPTIONS=$2

##### Evaluate CONCAT CAPTIONING #####
python ${GITHUB_DIR}/eval_reranking/concat_text_for_bleu.py ${OUTPUT_PATH}/pred.tkn $NUM_KEY_FRAMES_AND_CAPTIONS
python ${GITHUB_DIR}/eval_reranking/concat_text_for_bleu.py ${OUTPUT_PATH}/gold.tkn $NUM_KEY_FRAMES_AND_CAPTIONS

java -Xmx2G -jar ${METEOR_DIR}/meteor-*.jar ${OUTPUT_PATH}/pred.tkn.concat ${OUTPUT_PATH}/gold.tkn.concat -l en -norm > ${OUTPUT_PATH}/meteor.score.concat

python -m bleurt.score_files -candidate_file=${OUTPUT_PATH}/pred.tkn.concat -reference_file=${OUTPUT_PATH}/gold.tkn.concat -bleurt_checkpoint=${BLEURT_DIR}/bleurt/BLEURT-20 -scores_file=${OUTPUT_PATH}/bleurt.score.concat
