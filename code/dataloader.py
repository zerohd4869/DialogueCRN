import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np


class IEMOCAPDataset(Dataset):

    def __init__(self, path=None, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]


class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'train-valid':
            # we randomly extract 10% or 20% of the training data as validation split
            # since no pre-defined train/val split is provided in original IEMOCAP dataset
            self.keys = [x for x in self.trainIds] + [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]


class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, \
        self.trainVid, self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'user' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path=None, n_classes=-1, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, _ = pickle.load(open(path, 'rb'))

        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 3 else pad_sequence(dat[i], True) if i < 5 else dat[i].tolist() for i in dat]


class MELDRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):


        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        nine speakers: ['Chandler','The Interviewer','Ross','Phoebe','Joey','Rachel','Ross','Chandler','Mr. Geller']
        '''

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]
