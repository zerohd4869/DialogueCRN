# DialogueCRN
Source code for ACL-IJCNLP 2021 paper "[DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations](https://doi.org/10.18653/v1/2021.acl-long.547)".

## Quick Start

### Requirements
python==3.6.10          <br>
torch==1.4.0            <br>
torch-geometric==2.0.1  <br>
torch-scatter==2.0.4    <br>
sklearn==0.0            <br>
numpy==1.19.5           <br>
pandas==0.24.2          <br>

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
EXP_NO="dialoguecrn_v1"
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
EXP_NO="dialoguecrn_v1"
DATASET="meld"
WORK_DIR="${WORK_PATH}/DialogueCRN" # # your work path
DATA_DIR="${WORK_DIR}/data/${DATASET}/MELD_features_raw.pkl"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"

python -u ${WORK_DIR}/code/run_train_me.py   \
    --feature_type text --data_dir ${DATA_DIR} --output_dir ${OUT_DIR}  \
    --gamma 1.0 --step_s 3  --step_p 0  --lr 0.001 --l2 0.0002  --dropout 0.2 --base_layer 1

```

### Run examples
```bash
bash ./script/run_train_ie.sh
bash ./script/run_train_md.sh
```


## Result

Reproduced experiment results on th IEMOCAP and MELD datasets:



 | Model|IEMOCAP|  | | MELD| | |
 |:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
 | |Acc|w-F1| ma-F1 |Acc|w-F1| ma-F1|
 | TextCNN      |49.35|49.21|48.13|59.69|56.83|33.80|
 | bc-LSTM+Att  |56.32|56.19|54.84|57.50|55.90|34.84|
 | DialogueRNN  |63.03|62.50|60.66|59.54|56.39|32.93|
 | DialogueGCN  |64.02|63.65|63.42|59.46|56.77|34.05|
 | **DialogueCRN** |**66.73**|**66.66**|**67.25**|**61.26**|**58.48**|**35.69**|



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


