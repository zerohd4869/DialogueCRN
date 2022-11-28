# DialogueCRN
Source code for ACL-IJCNLP 2021 paper "[DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations](https://arxiv.org/pdf/2106.01978.pdf)".

In this work, we focus on emotion recognition in textual conversations (textual ERC). If you are interested in multimodal ERC, you can jump to our relevant work in [MM-DFN](https://github.com/zerohd4869/MM-DFN).

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
bash ./script/run_train_me.sh
```


## Results

Reproduced results of DialogueCRN on the IEMOCAP datasets:

|Model |Happy|Sad|Neutral|Angry|Excited|Frustrated|*Acc*|*Ma-F1*|*W-F1*|
|:----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|DialogueCRN (paper) |62.61|81.86|60.05|58.49|75.17|60.08|66.05|66.38|66.20|
|DialogueCRN + Multimodal |53.23|83.37|62.96|66.09|75.40|66.07|67.16|66.92|67.21|
|DialogueCRN + RoBERTa |54.28|81.34|69.57|62.09|67.33|64.22|67.39|66.47|67.53|


Reproduced results of DialogueCRN on the MELD datasets:


|Model |Neutral|Surprise|Fear|Sadness|Happy|Anger|Disgust|*Acc*|*Ma-F1*|*W-F1*|
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|DialogueCRN (paper) |76.62|49.69|-|24.37|53.70|44.91|-|60.73|35.51|58.39|
|DialogueCRN + Multimodal |77.01|50.10|-|26.63|52.77|45.15|-|61.11|35.95|58.67|
|DialogueCRN + RoBERTa |79.72|57.62|18.26|39.30|64.56|32.07|52.53|66.93|49.15|65.90|

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


