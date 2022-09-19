# DialogueCRN
Source code for ACL-IJCNLP 2021 paper "[DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations](https://doi.org/10.18653/v1/2021.acl-long.547)".

## Quick Start

### Requirements

* python 3.6.10          
* torch 1.4.0            
* torch-geometric 1.4.3
* torch-scatter 2.0.4
* scikit-learn 0.21.2
* CUDA 10.1

Install related dependencies:
```bash
pip install -r requirements.txt
```

### Dataset

The original datasets can be found at [IEMOCAP](https://sail.usc.edu/iemocap/), [SEMAINE](https://semaine-db.eu) and [MELD](https://github.com/SenticNet/MELD).

In this work, we focus on emotion recognition in textual conversations. Following previous works (bc-LSTM, DialogueRNN, DialogueGCN, et al.), raw features of textual modality are extracted by using TextCNN.

### Training/Testing

For training model on IEMOCAP dataset , you can refer to the following:
    
```bash
EXP_NO="dialoguecrn_base"
DATASET="iemocap"
WORK_DIR="${WORK_PATH}/DialogueCRN" # your work path
DATA_DIR="${WORK_DIR}/data/${DATASET}/IEMOCAP_features.pkl"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"

python -u ${WORK_DIR}/code/run_train_ie.py   \
    --feature_type text --data_dir ${DATA_DIR} --output_dir ${OUT_DIR}  \
    --gamma 0 --step_s 3  --step_p 4  --lr 0.0001 --l2 0.0002  --dropout 0.2 --base_layer 2
```

For training model on MELD dataset , you can refer to the following:

```bash
EXP_NO="dialoguecrn_base"
DATASET="meld"
WORK_DIR="${WORK_PATH}/DialogueCRN" # # your work path
DATA_DIR="${WORK_DIR}/data/${DATASET}/MELD_features_raw.pkl"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"

python -u ${WORK_DIR}/code/run_train_me.py   \
    --feature_type text --data_dir ${DATA_DIR} --output_dir ${OUT_DIR}  \
    --gamma 1.0 --step_s 3  --step_p 0  --lr 0.0005 --l2 0.0002  --dropout 0.2 --base_layer 1

```

### Run examples
```bash
bash ./script/run_train_ie.sh
bash ./script/run_train_md.sh
```


## Results

Results of DialogueCRN on the IEMOCAP datasets:

| **IEMOCAP**| | | | | | | | |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Happy|Sad|Neutral|Angry|Excited|Frustrated|Acc|Macro-F1|Weighted-F1|
|62.82|82.59|59.97|63.13|76.54|58.43|66.73|67.25|66.66|

Results of DialogueCRN on the MELD datasets:

| **MELD** | | | | | | | | |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Neutral|Surprise|Sadness|Happy|Anger|Fear/Disgust|Acc|Macro-F1|Weighted-F1|
|76.93|49.74|23.17|54.21|45.26|-|61.26|35.62|58.55|


# Citation
```
@inproceedings{DBLP:conf/acl/HuWH20,
  author    = {Dou Hu and
               Lingwei Wei and
               Xiaoyong Huai},
  title     = {DialogueCRN: Contextual Reasoning Networks for Emotion Recognition
               in Conversations},
  booktitle = {{ACL/IJCNLP} {(1)}},
  pages     = {7042--7052},
  publisher = {Association for Computational Linguistics},
  year      = {2021}
}
```


