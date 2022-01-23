from tqdm import tqdm
from torch import nn
import random
random.seed(1234)
import argparse
import torch
import random
import time
from torch.nn.utils.rnn import pack_padded_sequence
import math
from dataset import get_dataloaders
from models import Decoder_test, Encoder, Decoder, Seq2seq, Seq2seq_test
from utils import AverageMeter

# def train_epoch(model, train_dataloader, device, criterion, vocab_size, optimizer, args):
#     model.train()
#     losses = AverageMeter()
#     for idx, (imgs, trgs, lengths) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
#         optimizer.zero_grad()
#         imgs, trgs = imgs.to(device), trgs.to(device)
#         if random.random() < args.teacher_forcing:
#             targets = pack_padded_sequence(trgs, lengths, batch_first=True)[0]
#             outputs = model(imgs, trgs, lengths)
#             loss = criterion(outputs, targets)
#         else:
#             outputs = model.sample(imgs, trgs, lengths)
#             outputs = outputs.permute(1,0,2)[:,1:,:]
#             loss = criterion(outputs.contiguous().view(-1,vocab_size), trgs[:,1:].contiguous().view(-1,1).squeeze(1))
#         losses.update(loss.item(), imgs.shape[0])
#         loss.backward()
#         optimizer.step()
#     return losses.avg

def train_epoch(model, train_dataloader, device, criterion, vocab_size, optimizer, args):
    model.train()
    losses = AverageMeter()
    for idx, (imgs, trgs, lengths) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        imgs, trgs = imgs.to(device), trgs.to(device)
        targets = pack_padded_sequence(trgs[:,1:], lengths, batch_first=True)[0]
        if random.random() < args.teacher_forcing:
            outputs = model(imgs, trgs, lengths)
        else:
            outputs = model.sample(imgs, trgs, lengths)
        loss = criterion(outputs, targets)
        losses.update(loss.item(), imgs.shape[0])
        loss.backward()
        optimizer.step()
    return losses.avg


# def val_epoch(model, val_dataloader, device, criterion, vocab_size):
#     model.eval()
#     losses = AverageMeter()
#     with torch.no_grad():
#         for idx, (imgs, trgs, lengths) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
#             imgs, trgs = imgs.to(device), trgs.to(device)
#             if random.random() < args.teacher_forcing:
#                 targets = pack_padded_sequence(trgs, lengths, batch_first=True)[0]
#                 outputs = model(imgs, trgs, lengths)
#                 loss = criterion(outputs, targets)
#             else:
#                 outputs = model.sample(imgs, trgs, lengths)
#                 outputs = outputs.permute(1,0,2)[:,1:,:]
#                 loss = criterion(outputs.contiguous().view(-1,vocab_size), trgs[:,1:].contiguous().view(-1,1).squeeze(1))
#             losses.update(loss.item(), imgs.shape[0])
#     return losses.avg

def val_epoch(model, val_dataloader, device, criterion, vocab_size, args):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for idx, (imgs, trgs, lengths) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            imgs, trgs = imgs.to(device), trgs.to(device)
            targets = pack_padded_sequence(trgs[:,1:], lengths, batch_first=True)[0]
            if random.random() < args.teacher_forcing:
                outputs = model(imgs, trgs, lengths)
            else:
                outputs = model.sample(imgs, trgs, lengths)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), imgs.shape[0])
    return losses.avg

def train(args):

    train_dataloader, val_dataloader, vocab_size = get_dataloaders(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    encoder = Encoder(args.embedding_size).to(device)
    decoder = Decoder_test(vocab_size, args.embedding_size, args.hidden_size).to(device)
    seq2seq = Seq2seq_test(encoder, decoder, device)
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.embedding.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    best_val_loss = None

    try:
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            train_loss = train_epoch(seq2seq, train_dataloader, device, criterion, vocab_size, optimizer, args)
            val_loss = val_epoch(seq2seq, val_dataloader, device, criterion, vocab_size, args)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            train_loss, val_loss, math.exp(val_loss)))
            print('-' * 89)
            if not best_val_loss or val_loss < best_val_loss:
                print("saving model...")
                with open(f"showandtell_{args.dataset_ratio}.pt", 'wb') as f:
                    torch.save(seq2seq, f)
                best_val_loss = val_loss
            with open(f"latest_showandtell_{args.dataset_ratio}.pt", 'wb') as f:
                torch.save(seq2seq, f)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_size', type = int, default = 512)
    parser.add_argument('--hidden_size', type=int, default = 512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_dirs', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dataDir', type=str, default='../coco')
    parser.add_argument('--dataType', type=int, default=2017)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset_ratio', type=float, default=0.2)
    parser.add_argument('--teacher_forcing', type=float, default=0.5)

    args = parser.parse_args()

    train(args)