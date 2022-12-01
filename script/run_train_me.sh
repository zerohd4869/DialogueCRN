#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/DialogueCRN" # your work path
DATA_DIR="/DialogueCRN/data/meld/MELD_features_raw.pkl"


EXP_NO="dialoguecrn_base"
DATASET="meld"
echo "${EXP_NO}, ${DATASET}"

OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"
MODEL_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}/dialoguecrn_36.pkl"
LOG_PATH="${WORK_DIR}/logs/${DATASET}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

G="1"
S="0 1 2 3"

for g in ${G[@]}
do
    for ss in ${S[@]}
    do
        for sp in ${S[@]}
        do
        echo "gamma:${g}, step_s: ${ss}, step_p: ${sp}"
        python -u ${WORK_DIR}/code/run_train_me.py   \
            --status train  --feature_type text  --data_dir ${DATA_DIR}  --output_dir ${OUT_DIR}  --load_model_state_dir ${MODEL_DIR} \
            --gamma $g --step_s ${ss}  --step_p ${sp}  --lr 0.0005 --l2 0.0002  --dropout 0.2 --base_layer 1 --valid_rate 0.1   \
        >> ${LOG_PATH}/${EXP_NO}.out 2>&1

        done
    done
done


