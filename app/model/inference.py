import numpy as np
from pycocotools.coco import COCO
import random
random.seed(1234)
from dataset import Language

import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import os
from models import Encoder, Decoder_test, Seq2seq_test
import matplotlib.pyplot as plt

dataDir='../coco'
test_dataType='test2017'
test_image_path = f'{dataDir}/images/{test_dataType}'

transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])

embedding_size = 512
hidden_size = 512
num_layers = 2
num_dirs = 2
dropout = 0.1

dataDir='..'
train_dataType='train2017'
annFile='captions_{}.json'.format(train_dataType)
train_coco=COCO(annFile)

train_imgIds = train_coco.getImgIds()
train_length = len(train_imgIds)*0.5
train_annIds = train_coco.getAnnIds(imgIds=train_imgIds[:int(train_length)])
train_anns = train_coco.loadAnns(train_annIds)

vocab = Language(train_anns)
vocab.build_vocab()
vocab_size = len(vocab.word2idx)
vocab_size

device = torch.device("cpu")

encoder = Encoder(embedding_size)
decoder = Decoder_test(vocab_size, embedding_size, hidden_size)
seq2seq = Seq2seq_test(encoder, decoder, device).to(device)

seq2seq.load_state_dict(torch.load("dataset50.pt", map_location=device).state_dict())
seq2seq.eval()
print("finished_setup")

def inference(img):
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    sos = torch.tensor(2).view(1,1).to(device)
    with torch.no_grad():
        out = seq2seq.sample(img, sos, [25])
        _, top_ids = torch.max(out, dim=-1)
    sentence=[]
    for word in top_ids:
        if word==3:
            break
        sentence.append(vocab.idx2word[word])
    return ' '.join(sentence)