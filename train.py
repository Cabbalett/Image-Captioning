from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
import numpy as np
from pycocotools.coco import COCO
import numpy as np
from typing import List, Dict, Tuple, Sequence, Any
from collections import Counter, defaultdict, OrderedDict
from itertools import chain
import random
random.seed(1234)
from dataset import Language, NMTDataset
import argparse
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import time
import math
from dataset import get_dataloaders
from models import Encoder, Decoder, Seq2seq

def train_epoch(model, train_dataloader, device, criterion, vocab_size, optimizer):
    model.train()
    total_loss=0.
    for idx, (imgs, trgs) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        imgs, trgs = imgs.to(device), trgs.to(device)
        output = model(imgs,trgs)

        loss = criterion(output.view(-1,vocab_size), trgs.flatten())
        loss.backward()
        total_loss+=loss.item()
        optimizer.step()
    return total_loss / len(train_dataloader)

def val_epoch(model, val_dataloader, device, criterion, vocab_size):
    model.eval()
    total_loss=0.
    with torch.no_grad():
        for idx, (imgs, trgs) in tqdm(enumerate(val_dataloader)):
            imgs, trgs = imgs.to(device), trgs.to(device)
            output = model(imgs, trgs)
            total_loss+= criterion(output.view(-1, vocab_size), trgs.flatten()).item()
    return total_loss / len(val_dataloader)


def train(args):

    train_dataloader, val_dataloader, vocab_size = get_dataloaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(args.embedding_size)
    decoder = Decoder(vocab_size, args.embedding_size, args.hidden_size)
    seq2seq = Seq2seq(encoder, decoder, device)
    if torch.cuda.device_count() > 1:
        print("using multiple GPU")
        seq2seq = nn.DataParallel(seq2seq)
    seq2seq = seq2seq.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, seq2seq.parameters()),lr=0.001)

    try:
        for epoch in range(6):
            epoch_start_time = time.time()
            train_loss = train_epoch(seq2seq, train_dataloader, device, criterion, vocab_size, optimizer)
            print(train_loss)
            val_loss = val_epoch(seq2seq, val_dataloader, device, criterion, vocab_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embedding_size', type = int, default = 512)
    parser.add_argument('--hidden_size', type=int, default = 512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_dirs', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dataDir', type=str, default='../coco')
    parser.add_argument('--dataType', type=int, default=2017)
    parser.add_argument('--epochs', type=int, default=6)

    args = parser.parse_args()

    train(args)