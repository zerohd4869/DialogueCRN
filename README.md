# DialogueCRN
Source code for ACL-IJCNLP 2021 paper "[DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations](https://arxiv.org/pdf/2106.01978.pdf)".

In this work, we focus on emotion recognition in textual conversations (textual ERC). 
If you are interested in multimodal ERC, you can refer to a related work [MM-DFN](https://arxiv.org/pdf/2203.02385.pdf) ([code](https://github.com/zerohd4869/MM-DFN)).

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

Following previous works (bc-LSTM, DialogueRNN, DialogueGCN, et al.), raw utterance-level features of textual modality are extracted by TextCNN with Glove embedding.
The pre-extracted features can be found in [bc-LSTM](https://github.com/declare-lab/conv-emotion/tree/master/bc-LSTM-pytorch). If you want to train the features by yourself, you can refer to [DialogueGCN](https://github.com/declare-lab/conv-emotion/tree/master/DialogueGCN).

Besides, another alternative is to use BERT/RoBERTa to process text features, which will achieve better performance in most cases. You also can find the code and processed features with RoBERTa embedding in [COSMIC](https://github.com/declare-lab/conv-emotion/tree/master/COSMIC/feature-extraction).


### Training/Testing

**DialogueCRN with Glove features** (paper)

For training model on IEMOCAP dataset, you can refer to the following:
    
```bash
WORK_DIR="/DialogueCRN" # your work path
DATA_DIR="/DialogueCRN/data/iemocap/IEMOCAP_features.pkl" # your data path

EXP_NO="dialoguecrn_base"
DATASET="iemocap"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"

python -u ${WORK_DIR}/code/run_train_ie.py   \
    --status train  --feature_type text --data_dir ${DATA_DIR} --output_dir ${OUT_DIR}  \
    --gamma 0 --step_s 3  --step_p 4  --lr 0.0001 --l2 0.0002  --dropout 0.2 --base_layer 2
```

For training model on MELD dataset, you can refer to the following:

```bash
WORK_DIR="/DialogueCRN" # your work path
DATA_DIR="/DialogueCRN/data/meld/MELD_features_raw.pkl" # your data path

EXP_NO="dialoguecrn_base"
DATASET="meld"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"

python -u ${WORK_DIR}/code/run_train_me.py   \
    --status train  --feature_type text --data_dir ${DATA_DIR} --output_dir ${OUT_DIR}  \
    --gamma 1.0 --step_s 2  --step_p 0  --lr 0.0005 --l2 0.0002  --dropout 0.2 --base_layer 1 --valid_rate 0.1
```

Run examples:
```bash
# IEMOCAP dataset
bash ./script/run_train_ie.sh
# MELD dataset
bash ./script/run_train_me.sh
```


**DialogueCRN with RoBERTa features**

For training model on IEMOCAP dataset, you can refer to:

```bash
WORK_DIR="/DialogueCRN" # your work path
DATA_DIR="/DialogueCRN/data/iemocap/iemocap_features_roberta.pkl" # your data path

EXP_NO="dialoguecrn_bert_base"
DATASET="iemocap"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"

python -u ${WORK_DIR}/code/run_train_bert_ie.py   \
    --status train  --feature_type text  --data_dir ${DATA_DIR}  --output_dir ${OUT_DIR}    \
    --gamma 0  --step_s 3  --step_p 0  --lr 0.0001  --l2 0.0002  --dropout 0.2  --base_layer 2 --valid_rate 0.1
```

For training model on MELD dataset, you can refer to:
```bash
WORK_DIR="/DialogueCRN" # your work path
DATA_DIR="/DialogueCRN/data/meld/meld_features_roberta.pkl" # your data path

EXP_NO="dialoguecrn_bert_base"
DATASET="meld"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"

python -u ${WORK_DIR}/code/run_train_bert_me.py   \
    --status train  --feature_type text  --data_dir ${DATA_DIR}  --output_dir ${OUT_DIR}    \
    --gamma 1 --step_s 0  --step_p 1  --lr 0.0001 --l2 0.0002  --dropout 0.2 --base_layer 1 --use_valid_flag
```

Run examples:
```bash
# IEMOCAP dataset
bash ./script/run_train_bert_ie.sh 
# MELD dataset
bash ./script/run_train_bert_me.sh 

```

Note: The optimal hyper-parameters (e.g., the number of turns in Reasoning Modules) are selected according to the performance of validation set, with slight differences under different experimental configurations (i.e., the version of CUDA and PyTorch).


## Results

Results of DialogueCRN on the IEMOCAP dataset:

|Model |Happy|Sad|Neutral|Angry|Excited|Frustrated|*Acc*|*Weighted-F1*|
|:----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|**DialogueCRN (paper)** |62.61|81.86|60.05|58.49|75.17|60.08|66.05|66.20|
|DialogueCRN + Multimodal |53.23|83.37|62.96|66.09|75.40|66.07|67.16|67.21|
|DialogueCRN + RoBERTa |54.28|81.34|69.57|62.09|67.33|64.22|67.39|67.53|


Results of DialogueCRN on the MELD dataset:

|Model |Neutral|Surprise|Fear|Sad|Happy|Disgust|Anger|*Acc*|*Weighted-F1*|
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|**DialogueCRN (paper)** |76.62|49.69|-|24.37|53.70|-|44.91|60.73|58.39|
|DialogueCRN + Multimodal |77.01|50.10|-|26.63|52.77|-|45.15|61.11|58.67|
|DialogueCRN + RoBERTa |79.72|57.62|18.26|39.30|64.56|32.07|52.53|66.93|65.77|


## Citation
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


